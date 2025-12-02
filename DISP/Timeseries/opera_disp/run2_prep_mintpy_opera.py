#!/usr/bin/env python
"""Prepare and process OPERA DISP-S1 data for time series analysis with MintPy.

This script processes OPERA DISP-S1 products to generate:
1. Time series of cumulative displacement 
2. Average spatial/temporal coherence maps
3. Velocity maps and associated uncertainty estimates
4. Quality control masks and correction layers

The script handles parallel processing of large datasets and includes options for:
- Water masking and correction layers
- Reference point selection
- Short wavelength displacement extraction
- Quality control based on coherence thresholds

Example:
    run2_prep_mintpy_opera.py -u "outputs/*.nc" -m static_lyrs
        --geom-dir geometry -o mintpy_output --water-mask-file esa_world_cover_2021 
        --dem-file glo_30 --ref-lalo '29.692 -95.635' --n-workers 64 --chunk-size 50

Input Requirements:
    - DISP-S1 NetCDF files containing displacements
    - Static layer files with geometry information
    - Optional: DEM file, water mask, reference coordinates

Outputs:
    - timeseries.h5: Cumulative displacement time series
    - velocity.h5: Linear displacement rates
    - geometryGeo.h5: Geometry information in geographic coordinates
    - Multiple mask/coherence files for quality control

Dependencies:
    mintpy, gdal, rasterio, h5py, numpy, pandas, cartopy
    opera_utils (for handling OPERA-specific file formats)

Base code
Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi 
Author: Talib Oliver Cabrerra, Scott Staniewicz 

DISP-S1 Implementation
Author: Simran S Sangha, Jinwoo Kim
February, 2025
"""

# Standard library imports
import argparse
import glob
import itertools
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime as dt
from pathlib import Path
from typing import Sequence

import gc

# Third-party imports
import asf_search as asf
import h5py
import netCDF4
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import rasterio
import xarray as xr
from osgeo import gdal
from packaging.version import Version
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the src directory to sys.path
sys.path.append(str(Path(__file__).parent / "src"))

# Local application/library-specific imports
from mintpy.cli import (
    generate_mask,
    mask,
    dem_error,
)
from mintpy.reference_point import reference_point_attribute
from mintpy.utils import arg_utils, ptime, readfile, writefile
from mintpy.utils import utils as ut
from mintpy.utils.utils0 import azimuth2heading_angle, calc_azimuth_from_east_north_obs
from opera_utils import get_dates
from pst_dolphin_utils import (
    BackgroundRasterWriter,
    HDF5StackReader,
    create_external_files,
    datetime_to_float,
    estimate_velocity,
    full_suffix,
    get_raster_bounds,
    get_raster_crs,
    get_raster_gt,
    get_raster_xysize,
    load_gdal,
    process_blocks,
    warp_to_match,
    calculate_cumulative_displacement,
)
#from pst_ts_utils import calculate_cumulative_displacement
from tile_mate.stitcher import DATASET_SHORTNAMES

OPERA_DATASET_ROOT = './'

EXAMPLE = """example:
    run2_prep_mintpy_opera.py -u "outputs/*.nc" -m static_lyrs
        --geom-dir geometry -o mintpy_output --water-mask-file esa_world_cover_2021 
        --dem-file glo_30 --ref-lalo '29.692 -95.635' --n-workers 64 --chunk-size 50
"""

def _create_parser():
    parser = argparse.ArgumentParser(
        description="Prepare Sweets products for MintPy",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE,
    )

    parser.add_argument(
        "-u",
        "--unw-file-glob",
        type=str,
        default="./interferograms/unwrapped/*.unw.tif",
        help="path pattern of unwrapped interferograms (default: %(default)s).",
    )
    parser.add_argument(
        "-g",
        "--geom-dir",
        default="./geometry",
        help="Geometry directory (default: %(default)s).",
    )
    parser.add_argument(
        "-m",
        "--meta-file",
        type=str,
        help="GSLC metadata file or directory",
    )
    parser.add_argument(
        "-s",
        "--start-date",
        dest='startDate',
        default=None,
        help="remove/drop interferograms with date earlier than "
             "start-date in YYMMDD or YYYYMMDD format",
    )
    parser.add_argument(
        "-e",
        "--end-date",
        dest='endDate',
        default=None,
        help="remove/drop interferograms with date later than "
             "end-date in YYMMDD or YYYYMMDD format",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="./mintpy",
        help="output directory (default: %(default)s).",
    )
    parser.add_argument(
        "-r",
        "--range",
        dest="lks_x",
        type=int,
        default=1,
        help=(
            "number of looks in range direction, for multilooking applied after fringe"
            " processing.\nOnly impacts metadata. (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-a",
        "--azimuth",
        dest="lks_y",
        type=int,
        default=1,
        help=(
            "number of looks in azimuth direction, for multilooking applied after"
            " fringe processing.\nOnly impacts metadata. (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--water-mask-file",
        dest="water_mask_file",
        type=str,
        default=None,
        help="Specify either path to valid water mask, or download "
             f"using one of the following data sources: {DATASET_SHORTNAMES}",
    )
    parser.add_argument(
        "--dem-file",
        dest="dem_file",
        type=str,
        default=None,
        help="Specify either path to valid DEM, or download "
             "using one of the following data sources: srtm_v3, "
             "nasadem, glo_30, glo_90, glo_90_missing",
    )
    parser.add_argument(
        "--ref-lalo",
        dest="ref_lalo",
        type=str,
        default=None,
        help="Specify 'latitute longitude' of desired reference point. "
             "By default the pixel with the highest spatial coherence "
             "is selected",
    )
    parser.add_argument(
        "--zero-mask",
        dest="zero_mask",
        action="store_true",
        help="Mask all pixels with zero value in unw phase",
    )
    parser.add_argument(
        "--corr-lyrs",
        dest="corr_lyrs",
        action="store_true",
        help="Extract correction layers",
    )
    parser.add_argument(
        "--shortwvl-lyrs",
        dest="shortwvl_lyrs",
        action="store_true",
        help="Extract short wavelength layers",
    )
    parser.add_argument(
        "--tropo-correction",
        dest="tropo_correction",
        action="store_true",
        help="Apply tropospheric correction using HRRR weather model"
    )
    parser.add_argument(
        "--work-dir",
        dest="work_dir",
        type=str,
        default="./raider_intermediate_workdir",
        help="Working directory for tropospheric correction intermediates"
    )
    parser.add_argument(
        "--mask-layers",
        dest="mask_lyrs",
        action="store_true",
        help="Extract all mask layers",
    )
    parser.add_argument(
        "--load-all-layers",
        dest="load_all_lyrs",
        action="store_true",
        help="Extract all layers",
    )
    parser.add_argument(
        "--apply-mask",
        dest="apply_mask",
        action="store_true",
        help="Apply epoch based masking",
    )
    parser.add_argument(
        "--dem-error",
        dest="dem_error",
        action="store_true",
        help="Apply DEM-error correction",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of workers used for writing timeseries in parallel"
    )
    parser.add_argument(
        "--reliability-threshold",
        dest="reliability_threshold",
        type=float,
        default=0.9,
        help="Percentage of pixels across epoch in the recommended mask "
             "layer that must be present across epochs"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Chunk size for memory-efficient parallel processing:"
              "the larger it is, the faster it gets, but it also increases memory risk."
    )

    parser = arg_utils.add_subset_argument(parser, geo=True)

    return parser


def cmd_line_parse(iargs=None):
    """Create the command line parser"""
    parser = _create_parser()
    inps = parser.parse_args(args=iargs)

    # in case meta_file is input as wildcard
    inps.meta_file = sorted(glob.glob(inps.meta_file))[0]

    return inps


def build_array_in_chunks(arr, chunk_size=50, array_dtype=np.int8,
    threshold=None):
    """
    Chunk array to avoid memory error
    """
    # Create an output array of the same shape and target dtype
    result = np.empty_like(arr, dtype=array_dtype)

    # Iterate over the array in chunks along the first dimension
    for i in range(0, arr.shape[0], chunk_size):
        # Select a chunk
        chunk = arr[i:i + chunk_size]
        # Perform the operation on the chunk and store in the result array
        if threshold is None:
            result[i:i + chunk_size] = \
                (chunk != 0).astype(array_dtype)
        else:
            result[i:i + chunk_size] = \
                (chunk >= threshold).astype(array_dtype)
    
    return result


def create_reliability_mask(mask_file, meta, threshold_ratio=0.9):
    """
    Create a reliability mask by summing valid pixels across time and
    applying a threshold
    """

    # set output file paths
    out_dir = os.path.dirname(mask_file)
    timeseries_density_file = os.path.join(out_dir, 'timeseries_density.h5')
    reliability_threshold_perc = int(threshold_ratio * 100)
    recommended_mask_thres_file = os.path.join(out_dir,
        f'recommended_mask_{reliability_threshold_perc}thresh.h5')

    # compute pixel density arrays from recommended mask array
    if os.path.exists(mask_file):
        with h5py.File(mask_file, 'r') as f:
            # Read the timeseries data
            mask_timeseries = f['timeseries'][:]
            # Get the number of images
            num_images = mask_timeseries.shape[0] - 1
            # Sum up the valid pixels (1's) across time
            sum_valid = np.sum(mask_timeseries, axis=0)
            # Calculate the threshold number of valid observations required
            threshold = int(num_images * threshold_ratio)
            # Create the final reliability mask
            reliability_mask = build_array_in_chunks(sum_valid,
                threshold=threshold)
            # compute time series density
            timeseries_density = sum_valid / num_images
    else:
        ts_file = os.path.join(out_dir, 'timeseries.h5')
        with h5py.File(ts_file, 'r') as f:
            # Read the timeseries data
            mask_timeseries = f['timeseries'][:]
            # Get the number of images
            num_images = mask_timeseries.shape[0] - 1
            # convert to binary array tracking nodata values
            mask_timeseries = build_array_in_chunks(mask_timeseries)
            # Sum up the valid pixels (1's) across time
            sum_valid = np.nansum(mask_timeseries, axis=0)
            # Calculate the threshold number of valid observations required
            threshold = int(num_images * threshold_ratio)
            # Create the final reliability mask
            reliability_mask = build_array_in_chunks(sum_valid,
                threshold=threshold)
            # compute time series density
            timeseries_density = sum_valid / num_images

    # write arrays to file
    meta["UNIT"] = "1"
    # write time series density to file
    meta["FILE_TYPE"] = 'timeseriesdensity'
    writefile.write(timeseries_density, timeseries_density_file,
                    metadata=meta)

    # write reliability mask to file
    meta["DATA_TYPE"] = 'int8'
    meta["FILE_TYPE"] = 'mask'
    writefile.write(reliability_mask, recommended_mask_thres_file,
                    metadata=meta)
        
    return


def prepare_metadata(meta_file, int_file, geom_dir, nlks_x=1, nlks_y=1):
    """Get the metadata from the GSLC metadata file and the unwrapped interferogram."""
    print("-" * 50)

    cols, rows = get_raster_xysize(int_file)

    meta_compass = h5py.File(meta_file, "r")
    meta = {}

    geotransform = get_raster_gt(int_file)
    meta["LENGTH"] = rows
    meta["WIDTH"] = cols

    meta["X_FIRST"] = geotransform[0]
    meta["Y_FIRST"] = geotransform[3]
    meta["X_STEP"] = geotransform[1]
    meta["Y_STEP"] = geotransform[5]
    meta["X_UNIT"] = meta["Y_UNIT"] = "meters"

    crs = get_raster_crs(int_file)
    meta["EPSG"] = crs.to_epsg()

    if str(meta["EPSG"]).startswith('326'):
         meta["UTM_ZONE"] = str(meta["EPSG"])[3:] + 'N'
    else:
         meta["UTM_ZONE"] = str(meta["EPSG"])[3:] + 'S'

    if "/science" in meta_compass:
        root = "/science/SENTINEL1/CSLC"
        processing_ds = f"{root}/metadata/processing_information"
        burst_ds = f"{processing_ds}/s1_burst_metadata"
        if burst_ds not in meta_compass:
            burst_ds = f"{processing_ds}/input_burst_metadata"
    else:
        root = OPERA_DATASET_ROOT
        processing_ds = f"{root}/metadata/processing_information"
        burst_ds = f"{processing_ds}/input_burst_metadata"

    meta["WAVELENGTH"] = meta_compass[f"{burst_ds}/wavelength"][()]
    meta["RANGE_PIXEL_SIZE"] = meta_compass[f"{burst_ds}/range_pixel_spacing"][()]
    meta["AZIMUTH_PIXEL_SIZE"] = 14.1
    meta["EARTH_RADIUS"] = 6371000.0

    # get heading from azimuth angle
    geom_path = Path(geom_dir)
    file_to_path = {
        "los_east": geom_path / "los_east.tif",
        "los_north": geom_path / "los_north.tif",
    }
    dsDict = {}
    for dsName, fname in file_to_path.items():
        data = readfile.read(fname, datasetName=dsName)[0]
        data[data == 0] = np.nan
        dsDict[dsName] = data
    azimuth_angle, _, _ = get_azimuth_ang(dsDict)
    azimuth_angle = np.nanmean(azimuth_angle)
    heading = azimuth2heading_angle(azimuth_angle)
    meta["HEADING"] = heading

    t0 = dt.strptime(
        meta_compass[f"{burst_ds}/sensing_start"][()].decode("utf-8"),
        "%Y-%m-%d %H:%M:%S.%f",
    )
    t1 = dt.strptime(
        meta_compass[f"{burst_ds}/sensing_stop"][()].decode("utf-8"),
        "%Y-%m-%d %H:%M:%S.%f",
    )
    t_mid = t0 + (t1 - t0) / 2.0
    meta["CENTER_LINE_UTC"] = (
        t_mid - dt(t_mid.year, t_mid.month, t_mid.day)
    ).total_seconds()
    meta["HEIGHT"] = 750000.0
    meta["STARTING_RANGE"] = meta_compass[f"{burst_ds}/starting_range"][()]
    meta["PLATFORM"] = meta_compass[f"{burst_ds}/platform_id"][()].decode("utf-8")
    meta["ORBIT_DIRECTION"] = meta_compass[f"{root}/metadata/orbit/orbit_direction"][
        ()
    ].decode("utf-8")
    meta["ALOOKS"] = 1
    meta["RLOOKS"] = 1

    # apply optional user multilooking
    if nlks_x > 1:
        meta["RANGE_PIXEL_SIZE"] = str(float(meta["RANGE_PIXEL_SIZE"]) * nlks_x)
        meta["RLOOKS"] = str(float(meta["RLOOKS"]) * nlks_x)

    if nlks_y > 1:
        meta["AZIMUTH_PIXEL_SIZE"] = str(float(meta["AZIMUTH_PIXEL_SIZE"]) * nlks_y)
        meta["ALOOKS"] = str(float(meta["ALOOKS"]) * nlks_y)

    return meta


def _get_date_pairs(filenames):
    str_list = [Path(f).stem for f in filenames]
    basenames_noext = [str(f).replace(full_suffix(f), "") for f in str_list]

    date_pairs = []
    for i in basenames_noext:
        num_parts = i.split('_')
        if len(num_parts) == 9:
            date_pair = f'{num_parts[6][:8]}_{num_parts[7][:8]}'
            date_pairs.append(date_pair)

    return date_pairs


def get_azimuth_ang(dsDict):
    """Compute the azimuth angle from east/north coefficients"""
    east = dsDict["los_east"]
    north = dsDict["los_north"]
    azimuth_angle = calc_azimuth_from_east_north_obs(east, north)

    return azimuth_angle, east, north


def process_file(args):
    """Worker function to process a single file"""
    file_inc, reflyr_name, water_mask, ref_y, ref_x, mask_dict, phase2range, track_version = args
    
    try:
        # Handle recommended mask case
        if track_version <= Version('0.8') and reflyr_name == 'recommended_mask':
            data = np.ones_like(water_mask, dtype=np.byte)
        else:
            data = load_gdal(file_inc, masked=True)

        # Apply reference point correction
        if ref_y is not None and ref_x is not None:
            data -= np.nan_to_num(data[ref_y, ref_x])

        # Apply mask thresholds
        for dict_key in mask_dict.keys():
            mask_lyr = file_inc.replace(reflyr_name, dict_key)
            mask_thres = mask_dict[dict_key]
            mask_data = load_gdal(mask_lyr)
            if reflyr_name == 'recommended_mask':
                data[mask_data < mask_thres] = 0
            else:
                data[mask_data < mask_thres] = np.nan

        # Apply water mask
        data = data * water_mask

        # Handle unwrapped files
        if reflyr_name in ['unwrapped_phase', 'displacement']:
            data = np.nan_to_num(data)

        return data * phase2range

    except Exception as e:
        return f"Error processing {file_inc}: {str(e)}"


def save_stack(
    fname,
    ds_name_dict,
    meta,
    file_list,
    water_mask,
    date12_list,
    track_version,
    phase2range=1,
    ref_y=None,
    ref_x=None,
    unw_file=False,
    mask_dict={},
    n_workers=None,
    chunk_size=50  
):
    """Prepare h5 file for input stack of layers with parallel processing"""
    # Initialize HDF5 file
    writefile.layout_hdf5(fname, ds_name_dict, metadata=meta)
    
    # Get layer name
    reflyr_name = file_list[0].split(':')[-1]
    print(f"Writing data to HDF5 file {fname} with parallel processing...")
    
    # Use all available CPUs if not specified
    if n_workers is None:
        try:
            n_workers = len(psutil.Process().cpu_affinity())
        except:
            print('Using 10 Workers/CPU by default, this can be modify by adding the argument --n-workers XX')
            n_workers = 10

    # Process files in chunks to manage memory
    for chunk_start in range(0, len(file_list), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(file_list))
        chunk_files = file_list[chunk_start:chunk_end]
        
        # Prepare arguments for parallel processing
        process_args = [
            (f, reflyr_name, water_mask, ref_y, ref_x, mask_dict, phase2range, track_version) 
            for f in chunk_files
        ]
        
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_file, args): i 
                for i, args in enumerate(process_args)
            }
            
            for future in tqdm(as_completed(futures), 
                             total=len(futures), 
                             desc=f"Processing chunk {chunk_start//chunk_size + 1}"):
                idx = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, str) and result.startswith("Error"):
                        print(result)
                        continue
                    results.append((idx, result))
                except Exception as e:
                    print(f"Error processing file {idx}: {str(e)}")
                    continue

        # Sort results by original index
        results.sort(key=lambda x: x[0])
        
        # Write processed data to HDF5 in chunks
        with h5py.File(fname, "a") as f:
            for i, (_, data) in enumerate(results, chunk_start + 1):
                f["timeseries"][i] = data
                
            # Clear results after writing
            results.clear()
        
        # Force garbage collection
        gc.collect()

    # Set first acquisition to zero
    with h5py.File(fname, "a") as f:
        print("Setting value at the first acquisition to ZERO")
        f["timeseries"][0] = 0.0

    print(f"Finished writing to HDF5 file: {fname}")


def compute_displacement_parallel(date, date_list, water_mask, mask_dict,
    lyr_path, rows, cols, ref_y, ref_x, G, phase2range,
    apply_tropo_correction, work_dir, median_height):
    """Wrapper for parallel displacement computation"""

    result = calculate_cumulative_displacement(
        date, date_list, water_mask, mask_dict, lyr_path,  
        rows, cols, ref_y, ref_x, G, phase2range,
        apply_tropo_correction, work_dir, median_height)
    return result

def get_timeseries_parameters(prod_files):
    """Wrapper to return parameters for TS files"""

    # grab date list from the filename
    date12_list = _get_date_pairs(prod_files)
    num_file = len(prod_files)

    # Create a directed graph to represent date connections
    G = nx.DiGraph()
    
    # Add edges (measurements) to the graph
    date_pairs = [dl.split("_") for dl in date12_list]
    for (ref_date, sec_date), file in zip(date_pairs, prod_files):
        G.add_edge(ref_date, sec_date, file=file)

    # Get all unique dates
    date_list = sorted(set(itertools.chain.from_iterable(date_pairs)))
    num_date = len(date_list)

    # size info
    cols, rows = get_raster_xysize(prod_files[0])
    # baseline info
    pbase = np.zeros(num_date, dtype=np.float32)
    # define dataset structure
    dates = np.array(date_list, dtype=np.string_)
    ds_name_dict = {
        "date": [dates.dtype, (num_date,), dates],
        "bperp": [np.float32, (num_date,), pbase],
        "timeseries": [np.float32, (num_date, rows, cols), None],
    }

    return (
        G, date_pairs, date_list, date12_list, 
        num_date, cols, rows, ds_name_dict
    )


def generate_timeseries_h5(lyr_fname, lyr, ds_name_dict, meta, rows, cols,
    num_date, chunk_size, n_workers, date_list, water_mask, mask_dict,
    lyr_path, ref_y, ref_x, G, phase2range, apply_tropo_correction,
    work_dir, median_height):
    """Wrapper to generate TS h5 files"""

    if lyr != 'perpendicular_baseline':
        print(f"Writing data to HDF5 file {lyr_fname}")
        writefile.layout_hdf5(lyr_fname, ds_name_dict, metadata=meta)

        # Initialize with zeros
        with h5py.File(lyr_fname, "a") as f:
            print("Setting value at the first acquisition to ZERO")
            f["timeseries"][0] = np.zeros((rows, cols), dtype=np.float32)

    prog_bar = ptime.progressBar(maxValue=num_date)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_start in range(1, num_date, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_date)
            chunk_dates = date_list[chunk_start:chunk_end]
            
            # Initialize array just for this chunk
            chunk_timeseries = np.empty(
                (chunk_end - chunk_start, rows, cols), dtype=np.float32)
            
            # Submit jobs
            future_to_idx = {}  # Map futures to their indices
            for i, date in enumerate(chunk_dates):
                future = executor.submit(
                    compute_displacement_parallel,
                    date, date_list, water_mask, mask_dict, lyr_path,
                    rows, cols, ref_y, ref_x, G, phase2range,
                    apply_tropo_correction, work_dir, median_height
                )
                future_to_idx[future] = i

            # Process completed futures
            for future in as_completed(future_to_idx.keys()):
                idx = future_to_idx[future]
                try:
                    displacement = future.result()
                    if displacement is not None:
                        chunk_timeseries[idx] = displacement
                    prog_bar.update(chunk_start + idx,
                        suffix=date_list[chunk_start + idx])
                except Exception as e:
                    print(f"Error processing date {chunk_dates[idx]}: "
                          f"{str(e)}")

            # Write chunk to file
            if lyr == 'perpendicular_baseline':
                with h5py.File(lyr_fname, "a") as f:
                    chunk_timeseries = np.mean(chunk_timeseries, axis=(1, 2))
                    f["bperp"][chunk_start:chunk_end] = chunk_timeseries
            else:
                with h5py.File(lyr_fname, "a") as f:
                    f["timeseries"][chunk_start:chunk_end] = chunk_timeseries
                
            # Clear chunk data from memory
            chunk_timeseries = None
            gc.collect()

    prog_bar.close()
    print("finished writing to HDF5 file: {}".format(lyr_fname))

    return

def prepare_timeseries(
    outfile,
    unw_files,
    shortwvl_files,
    recmsk_files,
    track_version,
    metadata,
    last_indices,
    water_mask_file=None,
    ref_lalo=None,
    corr_lyrs=False,
    shortwvl_lyrs=False,
    apply_tropo_correction=False,
    median_height=50,
    work_dir=None,
    mask_lyrs=False,
    apply_mask=False,
    n_workers=16,
    chunk_size=25 
):
    """
    Prepare the timeseries file accounting for different reference dates
    in input files.
    
    The function now handles cases where input files might have different
    reference dates, calculating cumulative displacement by properly chaining
    measurements based on their date pairs.
    """
    print("-" * 50)
    print("preparing timeseries file: {}".format(outfile))

    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}

    # pass lyr to disp conversion factor, which depends on the product version
    sp_coh_lyr_name = 'interferometric_correlation'
    if track_version == Version('0.3'):
        disp_lyr_name = 'unwrapped_phase'
        phase2range = -1 * float(meta["WAVELENGTH"]) / (4.0 * np.pi)
    if track_version >= Version('0.4'):
        disp_lyr_name = 'displacement'
        phase2range = 1
    if track_version == Version('0.7'):
       sp_coh_lyr_name = 'estimated_spatial_coherence'
    if track_version >= Version('0.8'):
       sp_coh_lyr_name = 'estimated_phase_quality'

    if apply_tropo_correction and work_dir:
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(f'{work_dir}/orbits', exist_ok=True)
        os.makedirs(f'weather_files', exist_ok=True)

    # return TS parameters for displacement + correction layers
    (G, date_pairs, date_list, date12_list, num_date, cols, rows, 
     ds_name_dict) = get_timeseries_parameters(unw_files)

    # return TS parameters for short-wavelength layers
    (G_subset, date_pairs_subset, date_list_subset, date12_list_subset,
    num_date_subset, cols_subset, rows_subset,
    ds_name_dict_subset) = get_timeseries_parameters(shortwvl_files)

    # return TS parameters for recommended mask layers
    (G_recmsk, date_pairs_recmsk, date_list_recmsk, date12_list_recmsk,
    num_date_recmsk, cols_recmsk, rows_recmsk,
    ds_name_dict_recmsk) = get_timeseries_parameters(recmsk_files)

    # read water mask
    if water_mask_file is not None:
         water_mask = readfile.read(water_mask_file,
             datasetName='waterMask')[0]
    else:
        water_mask = np.ones((rows, cols), dtype=np.float32)

    # handle reference point
    coord = ut.coordinate(meta)
    if ref_lalo is not None:
        ref_lat, ref_lon = ref_lalo.split()
        ref_y, ref_x = coord.geo2radar(np.array(float(ref_lat)),
                                     np.array(float(ref_lon)))[0:2]
    else:
        coh_file = os.path.join(os.path.dirname(outfile),
            'avg_lyrs/estimatedSpatialCoherence.h5')
        coh = readfile.read(coh_file)[0]
        if water_mask.dtype.name == 'bool':
            coh[water_mask == False] = 0.0
        else:
            coh[water_mask == 0] = 0.0
        ref_y, ref_x = np.unravel_index(np.argmax(coh), coh.shape)
        del coh

    # update metadata
    ref_meta = reference_point_attribute(meta, y=ref_y, x=ref_x)
    meta.update(ref_meta)
    meta["FILE_TYPE"] = "timeseries"
    meta["UNIT"] = "m"

    # set dictionary that will be used to mask TS by specified thresholds
    # vestigial placeholder
    mask_dict = {}
    
    # loop through and write TS displacement and correction layers
    all_outputs = [outfile]
    correction_layers = []
    if corr_lyrs is True:
        correction_layers = ['ionospheric_delay',
                             'solid_earth_tide']
        if track_version < Version('0.8'):
            correction_layers.append('tropospheric_delay')

    # Handle short wavelength layers
    if shortwvl_lyrs is True and track_version >= Version('0.4'):
        lyr = 'short_wavelength_displacement'
        lyr_fname = os.path.join(os.path.dirname(outfile), f'{lyr}.h5')
        all_outputs.append(lyr_fname)

        # generate TS file
        generate_timeseries_h5(lyr_fname, lyr, ds_name_dict_subset, meta,
            rows_subset, cols_subset, num_date_subset, chunk_size, n_workers,
            date_list_subset, water_mask, mask_dict, lyr, None, None,
            G_subset, phase2range, apply_tropo_correction, work_dir,
            median_height)

    # Generate TS files for displacement and correction layers
    for ind, lyr in enumerate([disp_lyr_name] + correction_layers):
        if ind == 0:
            lyr_fname = all_outputs[0]
            lyr_path = f'{disp_lyr_name}'
        else:
            lyr_fname = os.path.join(os.path.dirname(outfile), f'{lyr}.h5')
            if lyr in correction_layers:
                lyr_path = f'/corrections/{lyr}'
            all_outputs.append(lyr_fname)

        # generate TS file
        generate_timeseries_h5(lyr_fname, lyr, ds_name_dict, meta, rows,
            cols, num_date, chunk_size, n_workers, date_list, water_mask,
            mask_dict, lyr_path, ref_y, ref_x, G, phase2range,
            apply_tropo_correction, work_dir, median_height)

        # pass bperp info to TS file
        if ind == 0:
            lyr = 'perpendicular_baseline'
            lyr_path = f'/corrections/{lyr}'
            generate_timeseries_h5(lyr_fname, lyr, ds_name_dict, meta, rows,
                cols, num_date, chunk_size, n_workers, date_list, water_mask,
                mask_dict, lyr_path, None, None, G, 1,
                False, work_dir, median_height)

    # Handle additional layers (correction layers, mask layers, etc.)
    mask_layers = ['recommended_mask']
    if mask_lyrs is True:
        mask_layers.extend(['connected_component_labels',
            'temporal_coherence', sp_coh_lyr_name])

    # Water mask layers available for version >= 0.8
    if track_version >= Version('0.8') and mask_lyrs is True:
        mask_layers.extend(['water_mask'])

    # Timeseries inversion residuals available for version >= 1.0
    if track_version >= Version('1.0') and mask_lyrs is True:
        mask_layers.extend(['timeseries_inversion_residuals'])
        phase2range = -1 * float(meta["WAVELENGTH"]) / (4.0 * np.pi)

    # need to manually build recommended mask in <=0.7 products
    # if v0.8, manually build up recommended mask because it is blank
    # within the product
    if track_version <= Version('0.8'):
        mask_dict['connected_component_labels'] = 1
        mask_dict['temporal_coherence'] = 0.6
        mask_dict[sp_coh_lyr_name] = 0.5

    # Write mask and correlation layers to file
    for lyr in mask_layers:
        lyr_fname = os.path.join(os.path.dirname(outfile), f'{lyr}.h5')
        if lyr != 'recommended_mask':
            lyr_paths = [i.replace(disp_lyr_name, lyr) for i in unw_files]

        # need to convert TS inversion from radians
        if lyr == 'timeseries_inversion_residuals':
            save_stack(lyr_fname, ds_name_dict, meta, lyr_paths,
                       water_mask, date12_list, track_version, phase2range,
                       mask_dict=mask_dict, n_workers=n_workers,
                       chunk_size=chunk_size)
        else:
            if lyr == 'recommended_mask':
                save_stack(lyr_fname, ds_name_dict_recmsk, meta, recmsk_files,
                       water_mask, date12_list_recmsk, track_version, 1,
                       mask_dict=mask_dict, n_workers=n_workers,
                       chunk_size=chunk_size)
            else:
                save_stack(lyr_fname, ds_name_dict, meta, lyr_paths,
                       water_mask, date12_list, track_version, 1,
                       mask_dict=mask_dict, n_workers=n_workers,
                       chunk_size=chunk_size)
        all_outputs.append(lyr_fname)

    # apply epoch-based masking
    if apply_mask:
        mskfile = os.path.join(os.path.dirname(outfile),
            'recommended_mask.h5')
        if track_version >= Version('0.8'):
            # define chunk sizes
            chunks = {'time':-1, 'y':512, 'x':512}

            # Define a dictionary that maps each mask index
            # to a list of corresponding time indices in outfile
            mask_to_time_mapping = {}
            start_idx = 0  # Start from the first time index
            for mask_idx, end_idx in enumerate(last_indices.values()):
                mask_to_time_mapping[mask_idx+1] = list(
                    range(start_idx, end_idx + 2)
                )
                start_idx = end_idx + 2  # Move to the next range

            # Open the datasets with xarray using defined chunks
            with xr.open_mfdataset(outfile, chunks=chunks) as ds_ts:
                with xr.open_mfdataset(mskfile, chunks=chunks) as ds_msk:
                    # Initialize an empty array to hold the expanded mask
                    expanded_mask = xr.zeros_like(ds_ts['timeseries'])

                    # Iterate over each mask layer
                    # and apply it to the corresponding time indices
                    for mask_idx, time_indices in (
                        mask_to_time_mapping.items()
                    ):
                        chunk_msk = ds_msk['timeseries'][mask_idx]
                        for time_step in time_indices:
                            indexer = dict(phony_dim_1=time_step)
                            expanded_mask.loc[indexer] = chunk_msk

                    # Apply the expanded mask to the timeseries dataset
                    tsstack_ts = ds_ts['timeseries'] * expanded_mask

            # Save the modified variable back to the HDF5 file
            with h5py.File(outfile, mode="r+") as h5file:
                for start_c in range(0, num_date, chunk_size):
                    end_c = min(start_c + chunk_size, num_date)
                    # Compute only the current chunk
                    chunk_ts = tsstack_ts.isel(
                        phony_dim_1=slice(start_c, end_c)
                    )
                    h5file['timeseries'][
                        start_c:end_c, :, :
                    ] = chunk_ts.values
                    
                    # Clear memory
                    chunk_ts = None
                    gc.collect()

    return all_outputs, ref_meta


def mintpy_prepare_geometry(outfile, int_file, geom_dir, metadata,
                            water_mask_file=None):
    """Prepare the geometry file."""
    print('-' * 50)
    print(f'preparing geometry file: {outfile}')

    geom_path = Path(geom_dir)
    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}
    meta["FILE_TYPE"] = "geometry"

    file_to_path = {
        'los_east': geom_path / 'los_east.tif',
        'los_north': geom_path / 'los_north.tif',
        'height': geom_path / 'height.tif',
        'shadowMask': geom_path / 'layover_shadow_mask.tif',
    }

    if water_mask_file:
        file_to_path['waterMask'] = water_mask_file

    dsDict = {}
    for dsName, fname in file_to_path.items():
        try:
            data = readfile.read(fname, datasetName=dsName)[0]
            if dsName not in ['shadowMask', 'waterMask']:
                data[data == 0] = np.nan
            dsDict[dsName] = data

            # write data to HDF5 file
        except FileNotFoundError as e:  # https://github.com/insarlab/MintPy/issues/1081
            print(f'Skipping {fname}: {e}')

    # Compute the azimuth and incidence angles from east/north coefficients
    azimuth_angle, east, north = get_azimuth_ang(dsDict)
    dsDict['azimuthAngle'] = azimuth_angle

    up = np.sqrt(1 - east**2 - north**2)
    incidence_angle = np.rad2deg(np.arccos(up))
    dsDict['incidenceAngle'] = incidence_angle

    # write out slant range distance
    slant_range = netCDF4.Dataset(int_file, keepweakref=True)
    slant_range = float(
        slant_range.groups['metadata']['slant_range_mid_swath'][0]
    )
    slant_range = np.full_like(incidence_angle,
        fill_value=slant_range, dtype=np.float32)
    dsDict['slantRangeDistance'] = slant_range

    writefile.write(dsDict, outfile, metadata=meta)
    return outfile


def prepare_average_stack(outfile, stack, lyr_name, file_type, metadata,
    water_mask=None, n_workers=None):
    """Average and export specified layers with parallel processing"""
    if n_workers is None:
        try:
            n_workers = len(psutil.Process().cpu_affinity())
        except:
             print('Using 10 Workers/CPU by default, this can be modify by adding the argument --n-workers XX')
             n_workers = 10            

    # Get the data variable and compute mean
    avg_data = getattr(stack, lyr_name)
    avg_data = avg_data.chunk({'time': -1, 'y': 512, 'x': 512})
    avg_data = avg_data.mean(dim='time', skipna=True)
    
    # Convert to numpy array for writing
    data = avg_data.compute(num_workers=n_workers).values

    if water_mask is not None:
        data *= water_mask

    meta = metadata.copy()
    meta["FILE_TYPE"] = file_type
    meta["UNIT"] = "1"

    # Create dataset dict for writefile.write
    datasetDict = {file_type: data}
    writefile.write(datasetDict, outfile, metadata=meta)
    
    return outfile


def prepare_stack(
    out_dir,
    product_files,
    unw_files,
    disp_lyr_name,
    track_version,
    metadata,
    water_mask_file=None,
    ref_lalo=None,
    mask_lyrs=False,
    n_workers=32
):
    """Prepare the input unw stack."""
    print("-" * 50)
    # copy metadata to meta
    meta = {key: value for key, value in metadata.items()}
    meta["FILE_TYPE"] = "ifgramStack"

    # get list of *.unw file
    num_pair = len(unw_files)

    conv_factor = 1
    # >=v0.4 increment in units of m, must be converted to phs
    sp_coh_lyr_name = 'interferometric_correlation'
    if track_version >= Version('0.4'):
        conv_factor = -1 * (4.0 * np.pi) / float(metadata["WAVELENGTH"])
    if track_version == Version('0.7'):
       sp_coh_lyr_name = 'estimated_spatial_coherence'
    if track_version >= Version('0.8'):
       sp_coh_lyr_name = 'estimated_phase_quality'

    print(f"number of unwrapped interferograms: {num_pair}")

    # get list of interferometric correlation layers
    cor_files = \
        [i.replace(disp_lyr_name, sp_coh_lyr_name) \
         for i in unw_files]
    print(f"number of correlation files: {len(cor_files)}")

    # get list of conncomp layers
    cc_files = \
        [i.replace(disp_lyr_name, 'connected_component_labels') \
         for i in unw_files]
    print(f"number of connected components files: {len(cc_files)}")

    if len(cc_files) != len(unw_files) or len(cor_files) != len(unw_files):
        print(
            "the number of *.unw and *.unw.conncomp or *.cor files are "
            "NOT consistent"
        )
        if len(unw_files) > len(cor_files):
            print("skip creating ifgramStack files.")
            return

        print("Keeping only cor files which match a unw file")
        unw_dates_set = set([tuple(get_dates(f)) for f in unw_files])
        cor_files = \
            [f for f in cor_files if tuple(get_dates(f)) in unw_dates_set]
        cc_files = \
            [f for f in cc_files if tuple(get_dates(f)) in unw_dates_set]

    # read water mask
    if water_mask_file is not None:
         water_mask = readfile.read(water_mask_file,
             datasetName='waterMask')[0]
    else:
        cols, rows = get_raster_xysize(unw_files[0])
        water_mask = np.ones((rows, cols), dtype=np.float32)

    # get date info: date12_list
    date12_list = _get_date_pairs(product_files)

    # set xarray dataframe
    disp_df = pd.DataFrame(
        [os.path.basename(product).split('.nc')[0].split('_') \
         for product in product_files],
        columns=['project', 'level', 'product', 'mode', 'frame_id',
                 'polarization', 'start_date', 'end_date', 'version',
                 'production_date'])
    disp_df['path'] = product_files
    disp_df['date12'] = date12_list
    disp_df['date1'] = [i.split('_')[0] for i in date12_list]
    disp_df['date2'] = [i.split('_')[1] for i in date12_list]
    disp_df['production_date'] = pd.to_datetime(disp_df['production_date'],
        format='%Y%m%dT%H%M%SZ') 
    disp_df['start_date'] = pd.to_datetime(disp_df['start_date'],
        format='%Y%m%dT%H%M%SZ')
    disp_df['end_date'] = pd.to_datetime(disp_df['end_date'],
        format='%Y%m%dT%H%M%SZ')

    # Load stack
    chunks={'time':-1, 'y':4096, 'x':4096}
    stack = xr.open_mfdataset(disp_df.path.to_list(), chunks=chunks)

    # loop through and create files for spatial coherence
    spcoh_fname = os.path.join(out_dir, 'estimatedSpatialCoherence.h5')
    prepare_average_stack(spcoh_fname, stack, sp_coh_lyr_name,
        'estimatedSpatialCoherence', meta, water_mask, n_workers=n_workers)

    # loop through and create files for temporal coherence
    tempcoh_files = 'temporal_coherence'
    tempcoh_fname = os.path.join(out_dir, 'temporalCoherence.h5')
    prepare_average_stack(tempcoh_fname, stack, tempcoh_files,
        'temporalCoherence', meta, water_mask, n_workers=n_workers)

    if water_mask_file is not None:
        # determine whether the defined reference point is valid
        if ref_lalo is not None:
            ref_lat, ref_lon = ref_lalo.split()
            # get value at coordinate
            atr = readfile.read_attribute(spcoh_fname)
            coord = ut.coordinate(atr)
            (ref_y,
             ref_x) = coord.geo2radar(np.array(float(ref_lat)),
                                      np.array(float(ref_lon)))[0:2]
            val_at_refpoint = water_mask[ref_y, ref_x]
            # exit if reference point is in masked area
            if val_at_refpoint == False or val_at_refpoint ==  0:
                raise Exception(f'Specified input --ref-lalo {ref_lalo} '
                                'not in masked region. Inspect output file'
                                f'{water_mask_file} to inform selection '
                                'of new point.')

    # extract non-default mask layers
    if mask_lyrs is True:
        # loop through and create files for persistent scatterer
        ps_files = 'persistent_scatterer_mask'
        ps_fname = os.path.join(out_dir, 'persistent_scatterer_mask.h5')
        prepare_average_stack(ps_fname, stack, ps_files,
            'persistentScatterer', meta, water_mask)

        # loop through and create files for connected components
        conn_fname = os.path.join(out_dir, 'connectedComponent.h5')
        prepare_average_stack(conn_fname, stack, cc_files,
            'connectedComponent', meta, water_mask)

    return


def main(iargs=None):
    """Run the preparation functions."""
    inps = cmd_line_parse(iargs)

    start_time = time.time()

    product_files = sorted(
        glob.glob(inps.unw_file_glob),
        key=lambda x: dt.strptime(
            x.split('_')[-3][:8], '%Y%m%d'
        )
    )

    # track layer export options
    if inps.load_all_lyrs is True:
        inps.mask_lyrs = True
        inps.tropo_correction = True
        inps.shortwvl_lyrs = True
        inps.corr_lyrs = True
        print('Extracting all optional layers')

    # filter input by specified dates
    if inps.startDate is not None:
        startDate = int(inps.startDate)
        filtered_files = []
        for i in product_files:
            sec_date = int(os.path.basename(i).split('_')[7][:8])
            if sec_date >= startDate:
                filtered_files.append(i)
        product_files = filtered_files

    if inps.endDate is not None:
        endDate = int(inps.endDate)
        filtered_files = []
        for i in product_files:
            sec_date = int(os.path.basename(i).split('_')[7][:8])
            if sec_date <= endDate:
                filtered_files.append(i)
        product_files = filtered_files

    # filter out duplicate products
    str_list = [Path(f).stem for f in product_files]
    basenames_noext = [str(f).replace(full_suffix(f), "") for f in str_list]
    filtered_dict = {}
    filtered_files = []
    for i in product_files:
        prod_basename = str(Path(i).stem)
        num_parts = prod_basename.split('_')
        # get date pairs
        prod_pair = f'{num_parts[6][:8]}_{num_parts[7][:8]}'
        # get production time
        production_time = dt.strptime(prod_basename.split('_')[-1],
            "%Y%m%dT%H%M%SZ")
        # update with most recent duplicate product
        if prod_pair in filtered_dict.keys():
            if production_time > filtered_dict[prod_pair]:
                filtered_dict[prod_pair] = production_time
                print(
                    "Rejecting older duplicate product "
                    f"{str(Path(filtered_files[-1]).stem)} for newer "
                    f"product {prod_basename}"
                )
                filtered_files[-1] = i
        else:
            filtered_dict[prod_pair] = production_time
            filtered_files.append(i)

    product_files = filtered_files
    date12_list = _get_date_pairs(product_files)
    print(f"Found {len(product_files)} unwrapped files")

    # track reference date change-over indices for each mini-stack
    # first extract ref date values
    refdate_list = [entry.split('_')[0] for entry in date12_list]
    # Dictionary to store last indices before ref date changes
    last_indices = {}
    current_date1 = refdate_list[0]
    for i, date1 in enumerate(refdate_list):
        if date1 != current_date1:
            # Store the last index before change
            last_indices[current_date1] = i - 1
            # Update to new date1
            current_date1 = date1

    # Capture the last group’s last index
    last_indices[current_date1] = len(refdate_list) - 1

    # get subset of short-wavelength files to sample for TS stack
    # first and last product for each mini-stack
    shortwvl_file_subset = [filtered_files[i] for i in last_indices.values()]
    # first pass first date
    shortwvl_file_subset = [filtered_files[0]]
    # loop through n-1 reference changes
    for i in list(last_indices.values())[:-1]:
        shortwvl_file_subset.append(filtered_files[i])
        shortwvl_file_subset.append(filtered_files[i+1])
    # account for last reference date which has just one mini-stack product
    shortwvl_file_subset.append(
        filtered_files[list(last_indices.values())[-1]]
    )

    # get subset of recommended mask files to sample for TS stack
    recmsk_file_subset = [filtered_files[i] for i in last_indices.values()]

    # track product version
    track_version = []
    for i in product_files:
        fname = os.path.basename(i)
        version_n = Version(fname.split('_')[-2].split('v')[1])
        track_version.append(version_n)

    # exit if multiple versions are found
    track_version = list(set(track_version))
    if len(track_version) > 1:
        raise Exception(f'Multiple file version increments ({track_version}) '
                        'found in specified input. Version increments are ' 
                        'not compatible. '
                        'delete the PDF.')

    # pass unw conversion factor, which depends on the product version
    track_version = track_version[0]
    if track_version == Version('0.3'):
        disp_lyr_name = 'unwrapped_phase'
    if track_version >= Version('0.4'):
        disp_lyr_name = 'displacement'

    # append appropriate NETCDF prefixes
    unw_files = \
        [f'NETCDF:"{i}":{disp_lyr_name}' for i in product_files]
    shortwvl_files = \
        [f'NETCDF:"{i}":short_wavelength_displacement' 
         for i in shortwvl_file_subset]
    recmsk_files = \
        [f'NETCDF:"{i}":recommended_mask' 
         for i in recmsk_file_subset]

    # get geolocation info
    static_dir = Path(inps.meta_file)
    geometry_dir = Path(inps.geom_dir)
    geometry_dir.mkdir(exist_ok=True)
    crs = get_raster_crs(unw_files[0])
    epsg = crs.to_epsg()
    out_bounds = get_raster_bounds(unw_files[0])

    # create water mask, if not specified
    Path(inps.out_dir).mkdir(exist_ok=True)
    if inps.water_mask_file is not None:
        if not Path(inps.water_mask_file).exists():
            inps.water_mask_file = create_external_files(inps.water_mask_file,
               unw_files[0], out_bounds, crs, inps.out_dir,
               maskfile=True)

    # create DEM, if not specified
    if inps.dem_file is not None:
        if not Path(inps.dem_file).exists():
            inps.dem_file = create_external_files(inps.dem_file,
               unw_files[0], out_bounds, crs, inps.out_dir,
               demfile=True)
    # check if mask file already generated through dolphin
    else:
        dem_file = geometry_dir.glob('*_DEM.tif')
        dem_file = [i for i in dem_file]
        if len(dem_file) > 0:
            inps.dem_file = dem_file[0]
            print(f"Found DEM file {inps.dem_file}")

    # check if geometry files exist
    geometry_files = ['los_east.tif', 'los_north.tif',
                      'layover_shadow_mask.tif']
    missing_files = [file for file in geometry_files
                     if not (geometry_dir / file).is_file()]
    if missing_files != []:
        missing_files_str = ', '.join(missing_files)
        raise FileNotFoundError(
            'The following geometry files are missing in the directory '
            f'{geometry_dir}: {missing_files_str}'
        )

    # check if height file exists
    height_file = geometry_dir / "height.tif"
    if not height_file.is_file():
        print(f"Generated {height_file} file")
        warp_to_match(
                    input_file=inps.dem_file,
                    match_file=unw_files[0],
                    output_file=height_file,
                    resample_alg="cubic",
                )

    # get median height for tropo estimate
    with rasterio.open(height_file) as src:
       height_arr = np.nan_to_num(src.read(1))
       valid_mask = (height_arr != 0)
       median_height = int(np.median(height_arr[valid_mask]))

    # check static layer naming convention
    allcaps_geometry = True
    static_files = sorted(Path(static_dir).glob("*STATIC_*.h5"))
    # capture alternate filename convention
    if static_files == []:
        allcaps_geometry = False

    # translate input options
    # metadata
    meta_file = Path(inps.meta_file)
    if meta_file.is_dir():
        # Search for the line of sight static_layers file
        try:
            # Grab the first one in in the directory
            if allcaps_geometry is True:
                meta_file = next(meta_file.rglob("*STATIC_*.h5"))
            else:
                meta_file = next(meta_file.rglob("static_*.h5"))
        except StopIteration:
            raise ValueError(f"No static layers file found in {meta_file}")

    meta = prepare_metadata(
        meta_file, unw_files[0], geom_dir=inps.geom_dir,
        nlks_x=inps.lks_x, nlks_y=inps.lks_y
    )

    # prepare geometry file
    geom_file = os.path.join(inps.out_dir, "geometryGeo.h5")
    mintpy_prepare_geometry(geom_file, product_files[0], geom_dir=inps.geom_dir,
        metadata=meta, water_mask_file=inps.water_mask_file)

    if hasattr(inps, 'n_workers') and inps.n_workers:
        ncpus = inps.n_workers
    else:
        try:
            ncpus = len(psutil.Process().cpu_affinity())  # number of available CPUs
        except Exception:
            print("Using 10 workers/CPUs by default. This can be modified with the argument --n-workers XX.")
            ncpus = 10

    # prepare mask layer outputs
    avg_dir = os.path.join(inps.out_dir, 'avg_lyrs')
    Path(avg_dir).mkdir(exist_ok=True)
    prepare_stack(
        out_dir=avg_dir,
        product_files=product_files,
        unw_files=unw_files,
        disp_lyr_name=disp_lyr_name,
        track_version=track_version,
        metadata=meta,
        water_mask_file=inps.water_mask_file,
        ref_lalo=inps.ref_lalo,
        mask_lyrs=inps.mask_lyrs,
        n_workers=ncpus
    )

    # prepare TS file
    og_ts_file = os.path.join(inps.out_dir, "timeseries.h5")
    all_outputs = [og_ts_file]
    ref_meta = None
    # time-series (if inputs are multi-reference and outputs are for single-reference)
    all_outputs, ref_meta = prepare_timeseries(
        outfile=og_ts_file,
        unw_files=unw_files,
        shortwvl_files=shortwvl_files,
        recmsk_files=recmsk_files,
        track_version=track_version,
        metadata=meta,
        last_indices=last_indices,
        water_mask_file=inps.water_mask_file,
        ref_lalo=inps.ref_lalo,
        corr_lyrs=inps.corr_lyrs,
        shortwvl_lyrs=inps.shortwvl_lyrs,
        apply_tropo_correction=inps.tropo_correction,
        median_height=median_height,
        work_dir=inps.work_dir if inps.tropo_correction else None,
        mask_lyrs=inps.mask_lyrs,
        apply_mask=inps.apply_mask,
        n_workers=ncpus,
        chunk_size=inps.chunk_size
    )

    # prepare recommended mask-based density and threshold mask
    recommended_mask_file = os.path.join(inps.out_dir, 'recommended_mask.h5')
    create_reliability_mask(recommended_mask_file, meta,
        threshold_ratio=inps.reliability_threshold)

    # generate velocity fit(s)
    ts_dict = {}
    ts_dict['velocity'] = og_ts_file

    # if short wvl stack, take temporal average
    if inps.shortwvl_lyrs is True:
        ts_dict['velocity_shortwvl'] = os.path.join(inps.out_dir,
            'short_wavelength_displacement.h5')

    # apply DEM-error correction
    if inps.dem_error is True:
        dem_error_file = os.path.join(inps.out_dir, 'demErr.h5')
        # run DEM-error correction script
        iargs = [og_ts_file, '-g', geom_file, '--num-worker', str(ncpus),
            '--dem-err-file', dem_error_file]
        dem_error.main(iargs)
        # pass dem error file for velocity fit
        ts_dict['velocity_demErr'] = os.path.join(inps.out_dir,
            "timeseries_demErr.h5")

    for vel_name, ts_name in ts_dict.items():
        # first set variables
        dolphin_ref_tif = os.path.join(inps.out_dir, 'dolphin_reference.tif')
        dolphin_vel_file = os.path.join(inps.out_dir, f'{vel_name}.tif')
        vel_file = os.path.join(inps.out_dir, f'{vel_name}.h5')
        keep_open = False
        dset_names = 'timeseries'
        num_threads = 6
        block_shape = (256, 256)

        # extract one product to serve as a reference file
        ds = gdal.Translate(dolphin_ref_tif, unw_files[0])
        ds = None

        # get list of times WRT to the reference time
        # and also pass the TS data
        with h5py.File(ts_name, 'r') as f:
            ts_date_list = f['date'][:]
            ts_data = f['timeseries'][:]

        x_arr = [
            dt.strptime(
                date.decode('utf-8'), '%Y%m%d'
            )
            for date in ts_date_list
        ]
        x_arr = datetime_to_float(x_arr)

        # initiate dolphin file object
        writer = BackgroundRasterWriter(dolphin_vel_file,
            like_filename=dolphin_ref_tif)

        # run dolphin velocity fitting algorithm in blocks
        def read_and_fit(
            readers: Sequence[HDF5StackReader],
            rows: slice, cols: slice
        ) -> tuple[np.ndarray, slice, slice]:

            # Only use the cor_reader if it's the same shape as the unw_reader
            if len(readers) == 2:
                unw_reader, cor_reader = readers
                unw_stack = unw_reader[:, rows, cols]
                weights = cor_reader[:, rows, cols]
                cor_threshold = 0.4
                weights[weights < cor_threshold] = 0
            else:
                unw_stack = readers[0][:, rows, cols]
                weights = None

            # Fit a line to each pixel with weighted least squares
            return (
                estimate_velocity(
                    x_arr=x_arr,
                    unw_stack=unw_stack,
                    weight_stack=weights
                ),
                rows,
                cols,
            )

        readers = [ts_data]
        process_blocks(
            readers=readers,
            writer=writer,
            func=read_and_fit,
            block_shape=block_shape,
            num_threads=num_threads,
        )

        writer.notify_finished()

        # delete temporary reference file
        os.remove(dolphin_ref_tif)

        # update metadata field
        start_date = date12_list[0].split('_')[0]
        end_date = date12_list[-1].split('_')[-1]
        meta["DATA_TYPE"] = 'float32'
        meta["DATE12"] =  start_date + '_' + end_date
        meta["FILE_PATH"] = ts_name
        meta["FILE_TYPE"] = 'velocity'
        meta["NO_DATA_VALUE"] = 'none'
        meta["PROCESSOR"] = 'dolphin'
        meta["REF_DATE"] = start_date
        meta["START_DATE"] = start_date
        meta["END_DATE"] = end_date
        meta["UNIT"] = 'm/year'
        # apply reference point to velocity file
        if ref_meta is not None:
            meta['REF_LAT'] = ref_meta['REF_LAT']
            meta['REF_LON'] = ref_meta['REF_LON']
            meta['REF_Y'] = ref_meta['REF_Y']
            meta['REF_X'] = ref_meta['REF_X']

        # initiate HDF5 file
        row = int(meta['LENGTH'])
        col = int(meta['WIDTH'])
        ds_name_dict = {
            "velocity": [np.float32, (row, col), None],
        }
        writefile.layout_hdf5(vel_file, ds_name_dict, metadata=meta)

        # writing data to HDF5 file
        print("writing data to HDF5 file {} with a mode ...".format(vel_file))
        with h5py.File(vel_file, "a") as f:
            vel_arr = gdal.Open(dolphin_vel_file).ReadAsArray()
            # Convert 0s to NaN
            vel_arr = np.where(vel_arr == 0, np.nan, vel_arr)
            # apply reference point
            if ref_meta is not None:
                ref_y = int(ref_meta['REF_Y'])
                ref_x = int(ref_meta['REF_X'])
                vel_arr -= np.nan_to_num(vel_arr[ref_y, ref_x])
            # write to file
            f["velocity"][:] = vel_arr

        del ts_data, readers
        print("finished writing to HDF5 file: {}".format(vel_file))

        # generate mask file from unw phase field
        if inps.zero_mask is True:
            msk_file = os.path.join(os.path.dirname(ts_name),
                'combined_msk.h5')
            iargs = [recommended_mask_file, '-o', msk_file, '--nonzero']
            generate_mask.main(iargs)

            # mask TS file, since reference_point adds offset back in masked field
            iargs = [ts_name, '--mask', msk_file]
            mask.main(iargs)

            # mask velocity file
            iargs = [vel_file, '--mask', msk_file]
            mask.main(iargs)

    elapsed_time = (time.time() - start_time) / 60
    print(f"Total processing time: {elapsed_time:.2f} minutes")
    print("Done.")
    return


if __name__ == "__main__":
    main(sys.argv[1:])
