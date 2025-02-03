#!/usr/bin/env python3
"""Utils for accessing PST DISP-S1 bursts products"""

import abc
import asf_search as asf
import logging
import os
from os import fspath, PathLike
from pathlib import Path
from concurrent.futures import (
    Executor,
    FIRST_EXCEPTION,
    Future,
    ThreadPoolExecutor,
    wait,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    NamedTuple,
    Optional,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)
from queue import Empty, Full, Queue
from threading import Event, Thread, main_thread
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime as dt, timedelta
import affine
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import jax.numpy as jnp
from jax import Array, jit, lax, vmap
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from tqdm.auto import tqdm
import rasterio
from osgeo import gdal, gdal_array, osr
from pyproj import CRS, Proj
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from collections.abc import Callable

from dem_stitcher.stitcher import stitch_dem
from tile_mate import get_raster_from_tiles
from tile_mate.stitcher import DATASET_SHORTNAMES

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 0.5

from builtins import Ellipsis

Index = Union[Ellipsis, slice, int]

if TYPE_CHECKING:
    PathLikeStr = PathLike[str]
else:
    PathLikeStr = PathLike

import sys

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

# Used for callable types
T = TypeVar("T")
P = ParamSpec("P")


class GeneralPath(Protocol):
    """A protocol to handle paths that can be either local or S3 paths."""

    def parent(self): ...

    def suffix(self): ...

    def resolve(self): ...

    def exists(self): ...

    def read_text(self): ...

    def __truediv__(self, other): ...

    def __str__(self) -> str: ...

    def __fspath__(self) -> str:
        return str(self)

PathOrStr = Union[str, PathLikeStr, GeneralPath]
Filename = PathOrStr


__all__ = [
    "BlockIndices",
    "BackgroundWorker",
    "BackgroundReader",
    "BackgroundWriter",
    "BackgroundRasterWriter",
    "DatasetReader",
    "DatasetWriter",
    "RasterWriter",
    "StackReader",
    "HDF5Reader",
    "HDF5StackReader",
    "BlockProcessor",
    "process_blocks",
    "iter_blocks",
]


# begin raster access/writing and multiprocessing functions
class DatasetWriter(Protocol):
    """An array-like interface for writing output datasets.

    `DatasetWriter` defines the abstract interface that types must conform to in order
    to be used by functions which write outputs in blocks.
    Such objects must export NumPy-like `dtype`, `shape`, and `ndim` attributes,
    and must support NumPy-style slice-based indexing for setting data..
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __setitem__(self, key: tuple[Index, ...], value: np.ndarray, /) -> None:
        """Write a block of data."""
        ...


class DatasetReader(Protocol):
    """An array-like interface for reading input datasets.

    `DatasetReader` defines the abstract interface that types must conform to in order
    to be read by functions which iterate in blocks over the input data.
    Such objects must export NumPy-like `dtype`, `shape`, and `ndim` attributes,
    and must support NumPy-style slice-based indexing.

    Note that this protocol allows objects to be passed to `dask.array.from_array`
    which needs `.shape`, `.ndim`, `.dtype` and support numpy-style slicing.
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __getitem__(self, key: tuple[Index, ...], /) -> ArrayLike:
        """Read a block of data."""
        ...


class StackReader(DatasetReader, Protocol):
    """An array-like interface for reading a 3D stack of input datasets.

    `StackReader` defines the abstract interface that types must conform to in order
    to be valid inputs to be read in functions like [dolphin.ps.create_ps][].
    It is a specialization of [DatasetReader][] that requires a 3D shape.
    """

    ndim: int = 3
    """int : Number of array dimensions."""

    shape: tuple[int, int, int]
    """tuple of int : Tuple of array dimensions."""

    def __len__(self) -> int:
        """Int : Number of images in the stack."""
        return self.shape[0]


def _mask_array(arr: np.ndarray, nodata_value: float | None) -> np.ndarray:
    """Mask an array based on a nodata value and return a regular array."""
    if np.isnan(nodata_value):
        # Mask invalid (NaN) values
        masked_arr = np.ma.masked_invalid(arr)
    else:
        # Mask values equal to the nodata_value
        masked_arr = np.ma.masked_equal(arr, nodata_value)
    
    # Replace masked values with 0
    return masked_arr.filled(0.)


@dataclass
class BaseStackReader(StackReader):
    """Base class for stack readers."""

    file_list : Sequence[Filename]
    readers: Sequence[DatasetReader]
    num_threads: int = 1
    nodata: Optional[float] = None

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        return _read_3d(key, self.readers, num_threads=self.num_threads)

    @property
    def shape_2d(self):
        return self.readers[0].shape

    @property
    def shape(self):
        return (len(self.file_list), *self.shape_2d)

    @property
    def dtype(self):
        return self.readers[0].dtype


class DummyProcessPoolExecutor(Executor):
    """Dummy ProcessPoolExecutor for to avoid forking for single_job purposes."""

    def __init__(self, max_workers: Optional[int] = None, **kwargs):  # noqa: D107
        self._max_workers = max_workers

    def submit(  # noqa: D102
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        future: Future = Future()
        result = fn(*args, **kwargs)
        future.set_result(result)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = True):  # noqa:D102
        pass

    def map(self, fn: Callable[P, T], *iterables, **kwargs):  # noqa: D102
        return map(fn, *iterables)


class BlockProcessor(Protocol):
    """Protocol for a block-wise processing function.

    Reads a block of data from each reader, processes it, and returns the result
    as an array-like object.
    """

    def __call__(
        self, readers: Sequence[StackReader], rows: slice, cols: slice
    ) -> tuple[ArrayLike, slice, slice]: ...


def process_blocks(
    readers: Sequence[StackReader],
    writer: DatasetWriter,
    func: BlockProcessor,
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
):
    """Perform block-wise processing over blocks in `readers`, writing to `writer`.

    Used to read and process a stack of rasters in parallel, setting up a queue
    of results for the `writer` to save.

    Note that the parallelism happens using a `ThreadPoolExecutor`, so `func` should
    be a function which releases the GIL during computation (e.g. using numpy).
    """
    shape = readers[0].shape[-2:]
    slices = list(iter_blocks(shape, block_shape=block_shape))

    pbar = tqdm(total=len(slices))

    # Define the callback to write the result to an output DatasetWrite
    def write_callback(fut: Future):
        data, rows, cols = fut.result()
        writer[..., rows, cols] = data
        pbar.update()

    Executor = ThreadPoolExecutor if num_threads > 1 else DummyProcessPoolExecutor
    futures: set[Future] = set()
    with Executor(num_threads) as exc:
        for rows, cols in slices:
            future = exc.submit(func, readers=readers, rows=rows, cols=cols)
            future.add_done_callback(write_callback)
            futures.add(future)

        while futures:
            done, futures = wait(futures, timeout=1, return_when=FIRST_EXCEPTION)
            for future in done:
                e = future.exception()
                if e is not None:
                    raise e


class StackReader(DatasetReader, Protocol):
    """An array-like interface for reading a 3D stack of input datasets.

    `StackReader` defines the abstract interface that types must conform to in order
    to be valid inputs to be read in functions like [dolphin.ps.create_ps][].
    It is a specialization of [DatasetReader][] that requires a 3D shape.
    """

    ndim: int = 3
    """int : Number of array dimensions."""

    shape: tuple[int, int, int]
    """tuple of int : Tuple of array dimensions."""

    def __len__(self) -> int:
        """Int : Number of images in the stack."""
        return self.shape[0]


@dataclass
class HDF5Reader(DatasetReader):
    """A Dataset in an HDF5 file.

    Attributes
    ----------
    filename : pathlib.Path | str
        Location of HDF5 file.
    dset_name : str
        Path to the dataset within the file.
    chunks : tuple[int, ...], optional
        Chunk shape of the dataset, or None if file is unchunked.
    keep_open : bool, optional (default False)
        If True, keep the HDF5 file handle open for faster reading.


    See Also
    --------
    BinaryReader
    RasterReader

    Notes
    -----
    If `keep_open=True`, this class does not store an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.
    If passing the `HDF5Reader` to multiple spawned processes, it is recommended
    to set `keep_open=False` .

    """

    filename: Path
    """pathlib.Path : The file path."""

    dset_name: str
    """str : The path to the dataset within the file."""

    nodata: Optional[float] = None
    """Optional[float] : Value to use for nodata pixels.

    If None, looks for `_FillValue` or `missing_value` attributes on the dataset.
    """

    keep_open: bool = False
    """bool : If True, keep the HDF5 file handle open for faster reading."""

    def __post_init__(self):
        filename = Path(self.filename)

        hf = h5py.File(filename, "r")
        dset = hf[self.dset_name]
        self.shape = dset.shape
        self.dtype = dset.dtype
        self.chunks = dset.chunks
        if self.nodata is None:
            self.nodata = dset.attrs.get("_FillValue", None)
            if self.nodata is None:
                self.nodata = dset.attrs.get("missing_value", None)
        if self.keep_open:
            self._hf = hf
            self._dset = dset
        else:
            hf.close()

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """Int : Number of array dimensions."""
        return len(self.shape)

    def __array__(self) -> np.ndarray:
        return self[:,]

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        if self.keep_open:
            data = self._dset[key]
        else:
            with h5py.File(self.filename, "r") as f:
                data = f[self.dset_name][key]
        return _mask_array(data, self.nodata) if self.nodata is not None else data


@dataclass
class HDF5StackReader(BaseStackReader):
    """A stack of datasets in an HDF5 file.

    See Also
    --------
    BinaryStackReader
    StackReader

    Notes
    -----
    If `keep_open=True`, this class stores an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.
    If passing the `HDF5StackReader` to multiple spawned processes, it is recommended
    to set `keep_open=False`.

    """

    @classmethod
    def from_file_list(
        cls,
        file_list : Sequence[Filename],
        dset_names: str | Sequence[str],
        keep_open: bool = False,
        num_threads: int = 1,
        nodata: Optional[float] = np.nan,
    ):
        """Create a HDF5StackReader from a list of files.

        Parameters
        ----------
        file_list : Sequence[Filename]
            List of paths to the files to read.
        dset_names : str | Sequence[str]
            Name of the dataset to read from each file.
            If a single string, will be used for all files.
        keep_open : bool, optional (default False)
            If True, keep the HDF5 file handles open for faster reading.
        num_threads : int, optional (default 1)
            Number of threads to use for reading.
        nodata : float, optional
            Manually set value to use for nodata pixels, by default None
            If None passed, will search for a nodata value in the file.

        Returns
        -------
        HDF5StackReader
            The HDF5StackReader object.

        """
        if isinstance(dset_names, str):
            dset_names = [dset_names] * len(file_list)

        readers = [
            HDF5Reader(Path(f), dset_name=dn, keep_open=keep_open, nodata=nodata)
            for (f, dn) in zip(file_list, dset_names)
        ]
        # Check if nodata values were found in the files
        nds = {r.nodata for r in readers}
        if len(nds) == 1:
            nodata = nds.pop()

        return cls(file_list, readers, num_threads=num_threads, nodata=nodata)


RasterT = TypeVar("RasterT", bound="RasterWriter")


class BackgroundWorker(abc.ABC):
    """Base class for doing work in a background thread.

    After instantiating an object, a client sends it work with the `queue_work`
    method and retrieves the result with the `get_result` method (hopefully
    after doing something else useful in between).  The worker remains active
    until `notify_finished` is called.  Subclasses must define the `process`
    method.

    Parameters
    ----------
    num_work_queue : int
        Max number of work items to queue before blocking, <= 0 for unbounded.
    num_results_queue : int
        Max number of results to generate before blocking, <= 0 for unbounded.
    store_results : bool
        Whether to store return values of `process` method.  If True then
        `get_result` must be called once for every `queue_work` call.
    timeout : float
        Interval in seconds used to check for finished notification once work
        queue is empty.

    Notes
    -----
    The usual caveats about Python threading apply.  It's typically a poor
    choice for concurrency unless the global interpreter lock (GIL) has been
    released, which can happen in IO calls and compiled extensions.

    """

    def __init__(
        self,
        num_work_queue=0,
        num_results_queue=0,
        store_results=True,
        drop_unfinished_results=False,
        timeout=_DEFAULT_TIMEOUT,
        name="BackgroundWorker",
    ):
        self.name = name
        self.store_results = store_results
        self.timeout = timeout
        self._finished_event = Event()
        self._work_queue = Queue(num_work_queue)
        if self.store_results:
            self._results_queue = Queue(num_results_queue)
        self._thread = Thread(target=self._consume_work_queue, name=name)
        self._thread.start()
        self._drop_unfinished_results = drop_unfinished_results

    def _consume_work_queue(self):
        while True:
            if not main_thread().is_alive():
                break

            logger.debug(f"{self.name} getting work")
            if self._finished_event.is_set():
                do_exit = self._drop_unfinished_results or (
                    self._work_queue.unfinished_tasks == 0
                )
                if do_exit:
                    break
                else:
                    # Keep going even if finished event is set
                    logger.debug(
                        f"{self.name} Finished... but waiting for work queue to empty,"
                        f" {self._work_queue.qsize()} items left,"
                        f" {self._work_queue.unfinished_tasks} unfinished"
                    )
            try:
                args, kw = self._work_queue.get(timeout=self.timeout)
                logger.debug(f"{self.name} processing")
                result = self.process(*args, **kw)
                self._work_queue.task_done()
                # Notify the queue that processing is done
                logger.debug(f"{self.name} got result")
            except Empty:
                logger.debug(f"{self.name} timed out, checking if done")
                continue

            if self.store_results:
                logger.debug(f"{self.name} saving result in queue")
                while True:
                    try:
                        self._results_queue.put(result, timeout=2)
                        break
                    except Full:
                        logger.debug(f"{self.name} result queue full, waiting...")
                        continue

    @abc.abstractmethod
    def process(self, *args, **kw):
        """User-defined task to operate in background thread."""

    def queue_work(self, *args, **kw):
        """Add a job to the work queue to be executed.

        Blocks if work queue is full.
        Same input interface as `process`.
        """
        if self._finished_event.is_set():
            msg = "Attempted to queue_work after notify_finished!"
            raise RuntimeError(msg)
        self._work_queue.put((args, kw))

    def get_result(self):
        """Get the least-recent value from the result queue.

        Blocks until a result is available.
        Same output interface as `process`.
        """
        while True:
            try:
                result = self._results_queue.get(timeout=self.timeout)
                self._results_queue.task_done()
                break
            except Empty as e:
                logger.debug(f"{self.name} get_result timed out, checking if done")
                if self._finished_event.is_set():
                    msg = "Attempted to get_result after notify_finished!"
                    raise RuntimeError(msg) from e
                continue
        return result

    def notify_finished(self, timeout=None):
        """Signal that all work has finished.

        Indicate that no more work will be added to the queue, and block until
        all work has been processed.
        If `store_results=True` also block until all results have been retrieved.
        """
        self._finished_event.set()
        if self.store_results and not self._drop_unfinished_results:
            self._results_queue.join()
        self._thread.join(timeout)

    def __del__(self):
        logger.debug(f"{self.name} notifying of exit")
        self.notify_finished()


class BackgroundWriter(BackgroundWorker):
    """Base class for writing data in a background thread.

    After instantiating an object, a client sends it data with the `queue_write`
    method.  The writer remains active until `notify_finished` is called.
    Subclasses must define the `write` method.

    Parameters
    ----------
    nq : int
        Number of write jobs that can be queued before blocking, <= 0 for
        unbounded.  Default is 1.
    timeout : float
        Interval in seconds used to check for finished notification once write
        queue is empty.

    """

    def __init__(self, nq=1, timeout=_DEFAULT_TIMEOUT, **kwargs):
        super().__init__(
            num_work_queue=nq,
            store_results=False,
            timeout=timeout,
            **kwargs,
        )

    # rename queue_work -> queue_write
    def queue_write(self, *args, **kw):
        """Add data to the queue to be written.

        Blocks if write queue is full.
        Same interfaces as `write`.
        """
        self.queue_work(*args, **kw)

    # rename process -> write
    def process(self, *args, **kw):
        self.write(*args, **kw)

    @property
    def num_queued(self):
        """Number of items waiting in the queue to be written."""
        return self._work_queue.qsize()

    @abc.abstractmethod
    def write(self, *args, **kw):
        """User-defined method for writing data."""


@dataclass
class RasterWriter(DatasetWriter, AbstractContextManager["RasterWriter"]):
    """A single raster band in a GDAL-compatible dataset containing one or more bands.

    `Raster` provides a convenient interface for using SNAPHU to unwrap ground-projected
    interferograms in raster formats supported by the Geospatial Data Abstraction
    Library (GDAL). It acts as a thin wrapper around a Rasterio dataset and a band
    index, providing NumPy-like access to the underlying raster data.

    Data access is performed lazily -- the raster contents are not stored in memory
    unless/until they are explicitly accessed by an indexing operation.

    `Raster` objects must be closed after use in order to ensure that any data written
    to them is flushed to disk and any associated file objects are closed. The `Raster`
    class implements Python's context manager protocol, which can be used to reliably
    ensure that the raster is closed upon exiting the context manager.
    """

    filename: Filename
    """str or Path : Path to the file to write."""
    band: int = 1
    """int : Band index in the file to write."""

    def __post_init__(self) -> None:
        # Open the dataset.
        self.dataset = rasterio.open(self.filename, mode="r+")

        # Check that `band` is a valid band index in the dataset.
        nbands = self.dataset.count
        if not (1 <= self.band <= nbands):
            errmsg = (
                f"band index {self.band} out of range: dataset contains {nbands} raster"
                " band(s)"
            )
            raise IndexError(errmsg)

        self.ndim = 2

    @classmethod
    def create(
        cls: type[RasterT],
        fp: Filename,
        width: int | None = None,
        height: int | None = None,
        dtype: DTypeLike | None = None,
        driver: str | None = None,
        crs: str | Mapping[str, str] | rasterio.crs.CRS | None = None,
        transform: rasterio.transform.Affine | None = None,
        *,
        like_filename: Filename | None = None,
        **kwargs: Any,
    ) -> RasterT:
        """Create a new single-band raster dataset.

        If another raster is passed via the `like` argument, the new dataset will
        inherit the shape, data-type, driver, coordinate reference system (CRS), and
        geotransform of the reference raster. Driver-specific dataset creation options
        such as chunk size and compression flags may also be inherited.

        All other arguments take precedence over `like` and may be used to override
        attributes of the reference raster when creating the new raster.

        Parameters
        ----------
        fp : str or path-like
            File system path or URL of the local or remote dataset.
        width, height : int or None, optional
            The numbers of columns and rows of the raster dataset. Required if `like` is
            not specified. Otherwise, if None, the new dataset is created with the same
            width/height as `like`. Defaults to None.
        dtype : data-type or None, optional
            Data-type of the raster dataset's elements. Must be convertible to a
            `numpy.dtype` object and must correspond to a valid GDAL datatype. Required
            if `like` is not specified. Otherwise, if None, the new dataset is created
            with the same data-type as `like`. Defaults to None.
        driver : str or None, optional
            Raster format driver name. If None, the method will attempt to infer the
            driver from the file extension. Defaults to None.
        crs : str, dict, rasterio.crs.CRS, or None; optional
            The coordinate reference system. If None, the CRS of `like` will be used, if
            available, otherwise the raster will not be georeferenced. Defaults to None.
        transform : rasterio.transform.Affine or None, optional
            Affine transformation mapping the pixel space to geographic space. If None,
            the geotransform of `like` will be used, if available, otherwise the default
            transform will be used. Defaults to None.
        like_filename : Raster or None, optional
            An optional reference raster. If not None, the new raster will be created
            with the same metadata (shape, data-type, driver, CRS/geotransform, etc) as
            the reference raster. All other arguments will override the corresponding
            attribute of the reference raster. Defaults to None.
        **kwargs : dict, optional
            Additional driver-specific creation options passed to `rasterio.open`.

        """
        if like_filename is not None:
            with rasterio.open(like_filename) as dataset:
                kwargs = dataset.profile | kwargs

        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if dtype is not None:
            kwargs["dtype"] = np.dtype(dtype)
        if driver is not None:
            kwargs["driver"] = driver
        if crs is not None:
            kwargs["crs"] = crs
        if transform is not None:
            kwargs["transform"] = transform

        # Always create a single-band dataset, even if `like` was part of a multi-band
        # dataset.
        kwargs["count"] = 1

        # Create the new single-band dataset.
        with rasterio.open(fp, mode="w+", **kwargs):
            pass

        return cls(fp, band=1)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.dataset.dtypes[self.band - 1])

    @property
    def height(self) -> int:
        """int : The number of rows in the raster."""  # noqa: D403
        return self.dataset.height  # type: ignore[no-any-return]

    @property
    def width(self) -> int:
        """int : The number of columns in the raster."""  # noqa: D403
        return self.dataset.width  # type: ignore[no-any-return]

    @property
    def shape(self):
        return self.height, self.width

    @property
    def closed(self) -> bool:
        """bool : True if the dataset is closed."""  # noqa: D403
        return self.dataset.closed  # type: ignore[no-any-return]

    def close(self) -> None:
        """Close the underlying dataset.

        Has no effect if the dataset is already closed.
        """
        if not self.closed:
            self.dataset.close()

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        self.close()

    def _window_from_slices(self, key: slice | tuple[slice, ...]) -> Window:
        if isinstance(key, slice):
            row_slice = key
            col_slice = slice(None)
        else:
            row_slice, col_slice = key

        return Window.from_slices(
            row_slice, col_slice, height=self.height, width=self.width
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}(dataset={self.dataset!r}, band={self.band!r})"

    def __setitem__(self, key: tuple[Index, ...], value: np.ndarray, /) -> None:
        with rasterio.open(
            self.filename,
            "r+",
        ) as dataset:
            if len(key) == 2:
                rows, cols = key
            elif len(key) == 3:
                _, rows, cols = _unpack_3d_slices(key)
            else:
                raise ValueError(
                    f"Invalid key for {self.__class__!r}.__setitem__: {key!r}"
                )
            try:
                window = Window.from_slices(
                    rows,
                    cols,
                    height=dataset.height,
                    width=dataset.width,
                )
            except rasterio.errors.WindowError as e:
                raise ValueError(f"Error creating window: {key = }, {value = }") from e
            return dataset.write(value, self.band, window=window)


class BackgroundRasterWriter(BackgroundWriter, DatasetWriter):
    """Class to write data to files in a background thread."""

    def __init__(
        self, filename: Filename, *, max_queue: int = 0, debug: bool = False, **kwargs
    ):
        super().__init__(nq=max_queue, name="Writer")
        if debug:
            #  background thread. Just synchronously write data
            self.notify_finished()
            self.queue_write = self.write  # type: ignore[assignment]

        if Path(filename).exists():
            self._raster = RasterWriter(filename)
        else:
            self._raster = RasterWriter.create(filename, **kwargs)
        self.filename = filename
        self.ndim = 2

    def write(self, key: tuple[Index, ...], value: np.ndarray):
        """Write out an ndarray to a subset of the pre-made `filename`.

        Parameters
        ----------
        key : tuple[Index,...]
            The key of the data to write.

        value : np.ndarray
            The block of data to write.

        """
        self._raster[key] = value

    def __setitem__(self, key: tuple[Index, ...], value: np.ndarray, /) -> None:
        self.queue_write(key, value)

    def close(self):
        """Close the underlying dataset and stop the background thread."""
        self._raster.close()
        self.notify_finished()

    @property
    def closed(self) -> bool:
        """bool : True if the dataset is closed."""  # noqa: D403
        return self._raster.closed

    @property
    def shape(self):
        return self._raster.shape

    @property
    def dtype(self) -> np.dtype:
        return self._raster.dtype

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        self.close()


@dataclass(frozen=True)
class BlockIndices:
    """Class holding slices for 2D array access."""

    row_start: int
    row_stop: Optional[int]  # Can be None if we want slice(0, None)
    col_start: int
    col_stop: Optional[int]

    @classmethod
    def from_slices(cls, row_slice: slice, col_slice: slice):
        return cls(
            row_start=row_slice.start,
            row_stop=row_slice.stop,
            col_start=col_slice.start,
            col_stop=col_slice.stop,
        )

    @property
    def row_slice(self) -> slice:
        return slice(self.row_start, self.row_stop)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_start, self.col_stop)

    # Allows user to unpack like (row_slice, col_slice) = block_index
    def __iter__(self):
        return iter((self.row_slice, self.col_slice))


def iter_blocks(
    arr_shape: tuple[int, int],
    block_shape: tuple[int, int],
    overlaps: tuple[int, int] = (0, 0),
    start_offsets: tuple[int, int] = (0, 0),
    end_margin: tuple[int, int] = (0, 0),
) -> Iterator[BlockIndices]:
    """Create a generator to get indexes for accessing blocks of a raster.

    Parameters
    ----------
    arr_shape : tuple[int, int]
        (num_rows, num_cols), full size of array to access
    block_shape : tuple[int, int]
        (height, width), size of blocks to load
    overlaps : tuple[int, int], default = (0, 0)
        (row_overlap, col_overlap), number of pixels to re-include from
        the previous block after sliding
    start_offsets : tuple[int, int], default = (0, 0)
        Offsets from top left to start reading from
    end_margin : tuple[int, int], default = (0, 0)
        Margin to avoid at the bottom/right of array

    Yields
    ------
    BlockIndices
        Iterator of BlockIndices, which can be unpacked into
        (slice(row_start, row_stop), slice(col_start, col_stop))

    Examples
    --------
        >>> list(iter_blocks((180, 250), (100, 100)))
        [BlockIndices(row_start=0, row_stop=100, col_start=0, col_stop=100), \
BlockIndices(row_start=0, row_stop=100, col_start=100, col_stop=200), \
BlockIndices(row_start=0, row_stop=100, col_start=200, col_stop=250), \
BlockIndices(row_start=100, row_stop=180, col_start=0, col_stop=100), \
BlockIndices(row_start=100, row_stop=180, col_start=100, col_stop=200), \
BlockIndices(row_start=100, row_stop=180, col_start=200, col_stop=250)]
        >>> list(map(tuple, iter_blocks((180, 250), (100, 100), overlaps=(10, 10))))
        [(slice(0, 100, None), slice(0, 100, None)), (slice(0, 100, None), \
slice(90, 190, None)), (slice(0, 100, None), slice(180, 250, None)), \
(slice(90, 180, None), slice(0, 100, None)), (slice(90, 180, None), \
slice(90, 190, None)), (slice(90, 180, None), slice(180, 250, None))]

    """
    total_rows, total_cols = arr_shape
    height, width = block_shape
    row_overlap, col_overlap = overlaps
    start_row_offset, start_col_offset = start_offsets
    last_row = total_rows - end_margin[0]
    last_col = total_cols - end_margin[1]

    if height is None:
        height = total_rows
    if width is None:
        width = total_cols

    # Check we're not moving backwards with the overlap:
    if row_overlap >= height and height != total_rows:
        msg = f"{row_overlap = } must be less than block height {height}"
        raise ValueError(msg)
    if col_overlap >= width and width != total_cols:
        msg = f"{col_overlap = } must be less than block width {width}"
        raise ValueError(msg)

    # Set up the iterating indices
    cur_row = start_row_offset
    cur_col = start_col_offset
    while cur_row < last_row:
        while cur_col < last_col:
            row_stop = min(cur_row + height, last_row)
            col_stop = min(cur_col + width, last_col)
            # yield (slice(cur_row, row_stop), slice(cur_col, col_stop))
            yield BlockIndices(cur_row, row_stop, cur_col, col_stop)

            cur_col += width
            if cur_col < last_col:  # dont bring back if already at edge
                cur_col -= col_overlap

        cur_row += height
        if cur_row < last_row:
            cur_row -= row_overlap
        cur_col = start_col_offset  # reset back to the starting offset


# begin utility functions
class Bbox(NamedTuple):
    """Bounding box named tuple, defining extent in cartesian coordinates.

    Usage:

        Bbox(left, bottom, right, top)

    Attributes
    ----------
    left : float
        Left coordinate (xmin)
    bottom : float
        Bottom coordinate (ymin)
    right : float
        Right coordinate (xmax)
    top : float
        Top coordinate (ymax)

    """

    left: float
    bottom: float
    right: float
    top: float


def create_external_files(lyr_name,
                          ref_file,
                          out_bounds,
                          crs,
                          out_dir='./',
                          maskfile=False,
                          demfile=False):
    """Generate mask and/or DEM file"""
    # get lat/lon bounds to generate mask and dem
    utm_proj = Proj(crs.to_wkt())
    e_lon, n_lat = utm_proj(out_bounds.left,
                            out_bounds.top,
                            inverse=True)
    w_lon, s_lat = utm_proj(out_bounds.right,
                            out_bounds.bottom,
                            inverse=True)
    geo_bounds = [e_lon, s_lat, w_lon, n_lat]
    # Get the affine transformation matrix
    with rasterio.open(ref_file) as src:
        reference_gt = src.transform
    resize_col, resize_row = get_raster_xysize(ref_file)
    #
    # download mask
    if maskfile:
        dat_arr, dat_prof = get_raster_from_tiles(geo_bounds,
                            tile_shortname=lyr_name)
        # fill permanent water body
        if lyr_name == 'esa_world_cover_2021':
            dat_arr[dat_arr == 80] = 0
            dat_arr[dat_arr != 0] = 1
        dat_arr = dat_arr.astype('byte')
        output_name = os.path.join(out_dir,
                      f'{lyr_name}_mask.tif')
        f_dtype = 'uint8'
        resampling_mode = Resampling.nearest
        print(f'mask file from source {lyr_name}\n')
    # download DEM
    if demfile:
        dst_area_or_point = 'Point'
        dst_ellipsoidal_height = True
        dat_arr, dat_prof = stitch_dem(geo_bounds,
                  dem_name=lyr_name,
                  dst_ellipsoidal_height=dst_ellipsoidal_height,
                  dst_area_or_point=dst_area_or_point)
        output_name = os.path.join(out_dir,
                      f'{lyr_name}_DEM.tif')
        f_dtype = 'float32'
        resampling_mode = Resampling.bilinear
        print(f'DEM file from source {lyr_name}\n')
    # resample
    with rasterio.open(output_name, 'w', 
                       height=resize_row, width=resize_col,
                       count=1,
                       dtype=f_dtype,
                       crs=crs,
                       transform=affine.Affine(*reference_gt)) as dst:
        reproject(source=dat_arr,
                  destination=rasterio.band(dst, 1),
                  src_transform=dat_prof['transform'],
                  src_crs=dat_prof['crs'],
                  dst_transform=reference_gt,
                  dst_crs=crs,
                  resampling=resampling_mode
                  )
    print(f'downloaded here: {output_name}\n')

    return output_name


def gdal_to_numpy_type(gdal_type: Union[str, int]) -> np.dtype:
    """Convert gdal type to numpy type."""
    if isinstance(gdal_type, str):
        gdal_type = gdal.GetDataTypeByName(gdal_type)
    return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type))


def get_raster_crs(filename) -> CRS:
    """Get the CRS from a file.

    Parameters
    ----------
    filename
        Path to the file to load.

    Returns
    -------
    CRS
        CRS.

    """
    ds = gdal.Open(fspath(filename))

    return CRS.from_wkt(ds.GetProjection())


def get_raster_bounds(
    filename = None, ds = None
) -> Bbox:
    """Get the (left, bottom, right, top) bounds of the image."""
    if ds is None:
        if filename is None:
            msg = "Must provide either `filename` or `ds`"
            raise ValueError(msg)
        ds = gdal.Open(fspath(filename))

    gt = ds.GetGeoTransform()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize

    left, top = _apply_gt(gt=gt, x=0, y=0)
    right, bottom = _apply_gt(gt=gt, x=xsize, y=ysize)

    return Bbox(left, bottom, right, top)


def _apply_gt(
    ds=None, filename=None, x=None, y=None, inverse=False, gt=None
) -> tuple[float, float]:
    """Read the (possibly inverse) geotransform, apply to the x/y coordinates."""
    if gt is None:
        if ds is None:
            ds = gdal.Open(fspath(filename))
            gt = ds.GetGeoTransform()
            ds = None
        else:
            gt = ds.GetGeoTransform()

    if inverse:
        gt = gdal.InvGeoTransform(gt)
    # Reference: https://gdal.org/tutorials/geotransforms_tut.html
    x = gt[0] + x * gt[1] + y * gt[2]
    y = gt[3] + x * gt[4] + y * gt[5]
    return x, y


def full_suffix(filename):
    """Get the full suffix of a filename, including multiple dots.

    Parameters
    ----------
    filename : str or Path
        path to file

    Returns
    -------
    str
        The full suffix, including multiple dots.

    Examples
    --------
        >>> full_suffix('test.tif')
        '.tif'
        >>> full_suffix('test.tar.gz')
        '.tar.gz'

    """
    fpath = Path(filename)
    return "".join(fpath.suffixes)


def get_raster_xysize(filename) -> tuple[int, int]:
    """Get the xsize/ysize of a GDAL-readable raster."""
    ds = gdal.Open(fspath(filename))
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    ds = None
    return xsize, ysize


def get_raster_gt(filename) -> list[float]:
    """Get the geotransform from a file.

    Parameters
    ----------
    filename
        Path to the file to load.

    Returns
    -------
    List[float]
        6 floats representing a GDAL Geotransform.

    """
    ds = gdal.Open(fspath(filename))
    return ds.GetGeoTransform()


def get_raster_driver(filename) -> str:
    """Get the GDAL driver `ShortName` from a file.

    Parameters
    ----------
    filename
        Path to the file to load.

    Returns
    -------
    str
        Driver name.

    """
    ds = gdal.Open(fspath(filename))
    return ds.GetDriver().ShortName


def get_raster_nodata(filename, band: int = 1):
    """Get the nodata value from a file.

    Parameters
    ----------
    filename
        Path to the file to load.
    band : int, optional
        Band to get nodata value for, by default 1.

    Returns
    -------
    Optional[float]
        Nodata value, or None if not found.

    """
    ds = gdal.Open(fspath(filename))
    return ds.GetRasterBand(band).GetNoDataValue()


def load_gdal(
    filename,
    *,
    band = None,
    subsample_factor = 1,
    overview = None,
    rows = None,
    cols = None,
    masked: bool = False,
) -> np.ndarray | np.ma.MaskedArray:
    """Load a gdal file into a numpy array.

    Parameters
    ----------
    filename : str or Path
        Path to the file to load.
    band : int, optional
        Band to load. If None, load all bands as 3D array.
    subsample_factor : int or tuple[int, int], optional
        Subsample the data by this factor. Default is 1 (no subsampling).
        Uses nearest neighbor resampling.
    overview: int, optional
        If passed, will load an overview of the file.
        Raster must have existing overviews, or ValueError is raised.
    rows : slice, optional
        Rows to load. Default is None (load all rows).
    cols : slice, optional
        Columns to load. Default is None (load all columns).
    masked : bool, optional
        If True, return a masked array using the raster's `nodata` value.
        Default is False.

    Returns
    -------
    arr : np.ndarray or np.ma.MaskedArray
        Array of shape (bands, y, x) or (y, x) if `band` is specified,
        where y = height // subsample_factor and x = width // subsample_factor.

    """
    ds = gdal.Open(fspath(filename))
    nrows, ncols = ds.RasterYSize, ds.RasterXSize

    if overview is not None:
        # We can handle the overviews most easily
        bnd = ds.GetRasterBand(band or 1)
        ovr_count = bnd.GetOverviewCount()
        if ovr_count > 0:
            idx = ovr_count + overview if overview < 0 else overview
            out = bnd.GetOverview(idx).ReadAsArray()
            bnd = ds = None
            return out
        logger.warning(f"Requested {overview = }, but none found for {filename}")

    # if rows or cols are not specified, load all rows/cols
    rows = slice(0, nrows) if rows in (None, slice(None)) else rows
    cols = slice(0, ncols) if cols in (None, slice(None)) else cols
    # Help out mypy:
    assert rows is not None
    assert cols is not None

    dt = gdal_to_numpy_type(ds.GetRasterBand(1).DataType)

    if isinstance(subsample_factor, int):
        subsample_factor = (subsample_factor, subsample_factor)

    xoff, yoff = int(cols.start), int(rows.start)
    row_stop = min(rows.stop, nrows)
    col_stop = min(cols.stop, ncols)
    xsize, ysize = int(col_stop - cols.start), int(row_stop - rows.start)
    if xsize <= 0 or ysize <= 0:
        msg = (
            f"Invalid row/col slices: {rows}, {cols} for file {filename} of size"
            f" {nrows}x{ncols}"
        )
        raise IndexError(msg)
    nrows_out, ncols_out = (
        ysize // subsample_factor[0],
        xsize // subsample_factor[1],
    )

    # Read the data, and decimate if specified
    resamp = gdal.GRA_NearestNeighbour
    if band is None:
        count = ds.RasterCount
        out = np.empty((count, nrows_out, ncols_out), dtype=dt)
        ds.ReadAsArray(xoff, yoff, xsize, ysize, buf_obj=out, resample_alg=resamp)
        if count == 1:
            out = out[0]
    else:
        out = np.empty((nrows_out, ncols_out), dtype=dt)
        bnd = ds.GetRasterBand(band)
        bnd.ReadAsArray(xoff, yoff, xsize, ysize, buf_obj=out, resample_alg=resamp)

    if not masked:
        return out
    # Get the nodata value
    nd = get_raster_nodata(filename)
    if nd is not None and np.isnan(nd):
        return np.ma.masked_invalid(out)
    else:
        return np.ma.masked_equal(out, nd)


def warp_to_match(
    input_file,
    match_file,
    output_file = None,
    resample_alg: str = "near",
    output_format = None,
) -> Path:
    """Reproject `input_file` to align with the `match_file`.

    Uses the bounds, resolution, and CRS of `match_file`.

    Parameters
    ----------
    input_file
        Path to the image to be reprojected.
    match_file
        Path to the input image to serve as a reference for the reprojected image.
        Uses the bounds, resolution, and CRS of this image.
    output_file
        Path to the output, reprojected image.
        If None, creates an in-memory warped VRT using the `/vsimem/` protocol.
    resample_alg: str, optional, default = "near"
        Resampling algorithm to be used during reprojection.
        See https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for choices.
    output_format: str, optional, default = None
        Output format to be used for the output image.
        If None, gdal will try to infer the format from the output file extension, or
        (if the extension of `output_file` matches `input_file`) use the input driver.

    Returns
    -------
    Path
        Path to the output image.
        Same as `output_file` if provided, otherwise a path to the in-memory VRT.

    """
    bounds = get_raster_bounds(match_file)
    crs_wkt = get_raster_crs(match_file).to_wkt()
    gt = get_raster_gt(match_file)
    resolution = (gt[1], gt[5])

    if output_file is None:
        output_file = f"/vsimem/warped_{Path(input_file).stem}.vrt"
        logger.debug(f"Creating in-memory warped VRT: {output_file}")

    if output_format is None and Path(input_file).suffix == Path(output_file).suffix:
        output_format = get_raster_driver(input_file)

    options = gdal.WarpOptions(
        dstSRS=crs_wkt,
        format=output_format,
        xRes=resolution[0],
        yRes=resolution[1],
        outputBounds=bounds,
        outputBoundsSRS=crs_wkt,
        resampleAlg=resample_alg,
    )
    gdal.Warp(
        fspath(output_file),
        fspath(input_file),
        options=options,
    )

    return Path(output_file)


def _ensure_slices(rows: Index, cols: Index) -> tuple[slice, slice]:
    def _parse(key: Index):
        if isinstance(key, int):
            return slice(key, key + 1)
        elif key is ...:
            return slice(None)
        else:
            return key

    return _parse(rows), _parse(cols)


def _unpack_3d_slices(key: tuple[Index, ...]) -> tuple[Index, slice, slice]:
    # Check that it's a tuple of slices
    if not isinstance(key, tuple):
        msg = "Index must be a tuple of slices."
        raise TypeError(msg)
    if len(key) not in (1, 3):
        msg = "Index must be a tuple of 1 or 3 slices."
        raise TypeError(msg)
    # If only the band is passed (e.g. stack[0]), convert to (0, :, :)
    if len(key) == 1:
        key = (key[0], slice(None), slice(None))
    # unpack the slices
    bands, rows, cols = key
    # convert the rows/cols to slices
    r_slice, c_slice = _ensure_slices(rows, cols)
    return bands, r_slice, c_slice


def _read_3d(
    key: tuple[Index, ...], readers: Sequence[DatasetReader], num_threads: int = 1
):
    bands, r_slice, c_slice = _unpack_3d_slices(key)

    if isinstance(bands, slice):
        # convert the bands to -1-indexed list
        total_num_bands = len(readers)
        band_idxs = list(range(*bands.indices(total_num_bands)))
    elif isinstance(bands, int):
        band_idxs = [bands]
    else:
        msg = "Band index must be an integer or slice."
        raise TypeError(msg)

    # Get only the bands we need
    if num_threads == 1:
        out = np.stack([readers[i][r_slice, c_slice] for i in band_idxs], axis=0)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(lambda i: readers[i][r_slice, c_slice], band_idxs)
        out = np.stack(list(results), axis=0)

    # TODO: Do i want a "keep_dims" option to not collapse singleton dimensions?
    return np.squeeze(out)


def datetime_to_float(dates) -> np.ndarray:
    """Convert a sequence of datetime objects to a float representation.

    Output units are in days since the first item in `dates`.

    Parameters
    ----------
    dates : Sequence[DateOrDatetime]
        List of datetime objects to convert to floats

    Returns
    -------
    date_arr : np.array 1D
        The float representation of the datetime objects

    """
    sec_per_day = 60 * 60 * 24
    date_arr = np.asarray(dates).astype("datetime64[s]")
    # Reference the 0 to the first date
    date_arr = date_arr - date_arr[0]
    return date_arr.astype(float) / sec_per_day


# begin velocity fitting functions
@jit
def estimate_velocity_pixel(x: ArrayLike, y: ArrayLike, w: ArrayLike) -> Array:
    """Estimate the velocity from a single pixel's time series.

    Parameters
    ----------
    x : np.array 1D
        The time values
    y : np.array 1D
        The unwrapped phase values
    w : np.array 1D
        The weights for each time value

    Returns
    -------
    velocity : np.array, 0D
        The estimated velocity in (unw unit) / year.

    """
    # Jax polyfit will grab the first *2* dimensions of y to solve in a batch
    return jnp.polyfit(x, y, deg=1, w=w.reshape(y.shape), rcond=None)[0]


@jit
def estimate_velocity(
    x_arr: ArrayLike, unw_stack: ArrayLike, weight_stack: ArrayLike | None
) -> Array:
    """Estimate the velocity from a stack of unwrapped interferograms.

    Parameters
    ----------
    x_arr : ArrayLike
        Array of time values corresponding to each unwrapped phase image.
        Length must match `unw_stack.shape[0]`.
    unw_stack : ArrayLike
        Array of unwrapped phase values at each pixel, shape=`(n_time, n_rows, n_cols)`.
    weight_stack : ArrayLike, optional
        Array of weights for each pixel, shape=`(n_time, n_rows, n_cols)`.
        If not provided, performs one batch unweighted linear fit.

    Returns
    -------
    velocity : np.array 2D
        The estimated velocity in (unw unit) / year calculated as 365.25 * rad/day.
        E.g. if the unwrapped phase is in radians, the velocity is in rad/year.

    """
    # TODO: weighted least squares using correlation?
    n_time, n_rows, n_cols = unw_stack.shape

    unw_pixels = unw_stack.reshape(n_time, -1)
    if weight_stack is None:
        # For jnp.polyfit(...), coeffs[0] is slope, coeffs[1] is the intercept
        velos = jnp.polyfit(x_arr, unw_pixels, deg=1, rcond=None)[0]
    else:
        # We use the same x inputs for all output pixels
        if unw_stack.shape != weight_stack.shape:
            msg = (
                "unw_stack and weight_stack must have the same shape,"
                f" got {unw_stack.shape} and {weight_stack.shape}"
            )
            raise ValueError(msg)

        weights_pixels = weight_stack.reshape(n_time, 1, -1)

        velos = vmap(estimate_velocity_pixel, in_axes=(None, -1, -1))(
            x_arr, unw_pixels, weights_pixels
        )
    # Currently `velos` is in units / day,
    days_per_year = 365.25
    return velos.reshape(n_rows, n_cols) * days_per_year


def calculate_cumulative_displacement(
    date, date_list, water_mask, mask_dict, lyr_name,
    rows, cols, ref_y, ref_x, G, phase2range,
    apply_tropo_correction, work_dir, median_height
):
    """
    Calculate cumulative displacement up to given date using shortest path
    """
    if date == date_list[0]:
        return np.zeros((rows, cols), dtype=np.float32)
    
    # Find shortest path from first date to current date
    try:
        # Finding all paths (files) in the shortest path from
        # the first date to given date
        path = nx.shortest_path(G, source=date_list[0], target=date)
    except nx.NetworkXNoPath:
        print(f"Warning: No path found to date {date}")
        return None

    # Calculate cumulative displacement along the path
    cumulative = np.zeros((rows, cols), dtype=np.float32)
    print(f'\ndate for calculating cumulative displacement: {date}')
    print(f'shortest path from {date_list[0]} to {date}')
    for i in range(len(path)-1):
        ref_date, sec_date = path[i], path[i+1]
        file = G[ref_date][sec_date]['file']	# File in the shortest path
        reflyr_name = file.split(':')[-1]
        file = file.replace(reflyr_name, lyr_name)
        print(f'{i} {ref_date} to {sec_date}: {file}')	

        # Read displacement data
        data = load_gdal(file, masked=True)
        data *= phase2range
        data = data.astype(np.float32)  # Convert to numeric type

        # Apply tropospheric correction if requested
        if apply_tropo_correction and work_dir:
            print(f"\nApplying tropospheric correction to pair {ref_date}-{sec_date}")
            
            # Extract original NetCDF filename from the GDAL path string
            # Format is 'NETCDF:"path/to/file.nc":displacement'
            nc_file = file.split('"')[1]
            print('nc file: ', nc_file)
            
            try:
                # Read parameters needed for tropo correction
                with h5py.File(nc_file, 'r') as nc:
                    track_number = nc['identification']['track_number'][()]
                    bounding_polygon = nc['identification']['bounding_polygon'][()].decode()
                    ref_datetime = dt.strptime(nc['identification']['reference_datetime'][()].decode(),
                                                '%Y-%m-%d %H:%M:%S.%f')
                    sec_datetime = dt.strptime(nc['identification']['secondary_datetime'][()].decode(),
                                                '%Y-%m-%d %H:%M:%S.%f')
                    spatial_ref_attrs = nc['spatial_ref'].attrs
                    crs_wkt = spatial_ref_attrs['crs_wkt']   
                    epsg_code = crs_wkt.split('ID["EPSG",')[-1].split(']')[0]
                    epsg_str = f'EPSG:{epsg_code}'
                    GeoTransform = spatial_ref_attrs['GeoTransform']
                    frame_id = nc['identification']['frame_id'][()]
                    frame_id = 'F' + str(frame_id).zfill(5)
                    mission_id = nc['identification']['mission_id'][()]
                    ref_date = ref_datetime.strftime('%Y%m%d')  # YYYYMMDD
                    sec_date = sec_datetime.strftime('%Y%m%d')
                    if 'unwrapper_mask' in nc.keys():
                        unwrapper_mask = nc['unwrapper_mask'][:]

                # Setup parameters for tropospheric correction
                params = {
                    'track_number': track_number,
                    'bounding_polygon': bounding_polygon,
                    'ref_datetime': ref_datetime,
                    'sec_datetime': sec_datetime,
                    'GeoTransform': GeoTransform,
                    'epsg' : epsg_str,
                    'median_height' : median_height,
                    'mission_id' : mission_id,
                    'height' : data.shape[0],
                    'width' : data.shape[1],
                }

                # Calculate and apply tropospheric correction
                calculated_tropo_delay = calculate_tropospheric_delay(params, work_dir)     # unit: meter
                
                ### code to plot the tropospheric correction
                _data = data.copy()
                if 'unwrapper_mask' in locals():
                    _data[unwrapper_mask==0.] = np.nan
                fig, ax = plt.subplots(1, 3, figsize=(15,10))
                im0 = ax[0].imshow(_data, cmap='RdBu')
                ax[0].set_title(f'Displacement before tropo correction \n{frame_id} {ref_date}-{sec_date}')
                ax[0].axis('off')
                plt.colorbar(im0, ax=ax[0], label='LOS (m)', shrink=0.2)
                im1 = ax[1].imshow(calculated_tropo_delay, cmap='RdBu')
                ax[1].set_title(f'Tropospheric delay \n{frame_id} {ref_date}-{sec_date}')
                ax[1].axis('off')
                plt.colorbar(im1, ax=ax[1], label='LOS (m)',shrink=0.2)
                im2 = ax[2].imshow(_data - calculated_tropo_delay, cmap='RdBu')
                ax[2].set_title(f'Displacement after tropo correction \n{frame_id} {ref_date}-{sec_date}')
                ax[2].axis('off')
                plt.colorbar(im2, ax=ax[2], label='LOS (m)', shrink=0.2)
                plt.tight_layout()
                fig.savefig(f'{work_dir}/tropo_corrected_displacement_{frame_id}_Raytracing_{ref_date}_{sec_date}.png', dpi=300, bbox_inches='tight')
                plt.close('all')
                del _data
                if 'unwrapper_mask' in locals():
                    del unwrapper_mask

                data -= calculated_tropo_delay      # unit: meter 

            except Exception as e:
                print(f"Warning: Tropospheric correction failed for {ref_date}_{sec_date}: {str(e)}")
                print("Continuing with uncorrected data...")

        # Apply reference point correction
        if ref_y is not None and ref_x is not None:
            data -= np.nan_to_num(data[ref_y, ref_x])
        
        # mask by specified dict of thresholds 
        for dict_key in mask_dict.keys():
            mask_lyr = file.replace(lyr_name, dict_key)
            mask_thres = mask_dict[dict_key]
            mask_data = load_gdal(mask_lyr)
            data[mask_data < mask_thres] = np.nan

        # Apply water mask
        data *= water_mask

        # Add to cumulative displacement
        cumulative += data 

    # convert nans to 0
    # necessary to avoid errors with MintPy velocity fitting
    return np.nan_to_num(cumulative)


def find_sentinel1_sensor(wkt_polygon, track_number, start_date):
    """
    Search for Sentinel-1 scenes and return the sensor (S1A or S1B)
    of the first found scene.
    
    Args:
        wkt_polygon (str): WKT representation of the area of interest
        track_number (int): track number of Sentinel-1
        start_date (str): Start date in 'YYYYMMDD' format
    
    Returns:
        str: Sensor name ('S1A' or 'S1B') or None if no scenes found
    """
    try:
        # Convert dates to datetime objects
        start = dt.strptime(start_date, '%Y%m%d')
        end = start + timedelta(days=1)

        results = asf.search(
            platform=[asf.PLATFORM.SENTINEL1],
            relativeOrbit=[track_number],
            processingLevel=[asf.PRODUCT_TYPE.SLC],
            start=start,
            end=end,
            maxResults=5,
            intersectsWith=wkt_polygon
        )

        if len(results) > 0:
            # Get the first scene's properties
            first_scene = results[0]
            
            # Extract platform name (Sentinel-1A or Sentinel-1B)
            sensor = first_scene.properties['platform']

            sensor_mapping = {
                "Sentinel-1A": "S1A",
                "Sentinel-1B": "S1B"
            }

            return sensor_mapping[sensor]
        else:
            print("No Sentinel-1 scenes found for the specified criteria")
            return None
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def get_bounds_from_geotransform(geotransform, width, height):
    """
    Get the corner coordinates from a GeoTransform
    """
    # GeoTransform format: (top_left_x, pixel_width, x_rotation, top_left_y, y_rotation, pixel_height)
    gt = [float(x) for x in geotransform.strip("'").split()]
    
    # Calculate corner coordinates in the original projection
    corners = [
        (gt[0], gt[3]),  # Top-left
        (gt[0] + width * gt[1], gt[3]),  # Top-right
        (gt[0], gt[3] + height * gt[5]),  # Bottom-left
        (gt[0] + width * gt[1], gt[3] + height * gt[5])  # Bottom-right
    ]
    return corners


def transform_coords(coords, source_epsg, target_epsg):
    """
    Transform coordinates from source EPSG to target EPSG
    """
    # Create coordinate transformation
    source = osr.SpatialReference()
    source.ImportFromEPSG(int(source_epsg.split(':')[1]))
    
    target = osr.SpatialReference()
    target.ImportFromEPSG(int(target_epsg.split(':')[1]))
    
    transform = osr.CoordinateTransformation(source, target)
    
    # Transform all coordinates
    transformed_coords = []
    for x, y in coords:
        point = transform.TransformPoint(x, y)
        transformed_coords.append((point[1], point[0])) 
    return transformed_coords

### EOF
