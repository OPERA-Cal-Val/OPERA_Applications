import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# --- Standard Library ---
import os
import sys
import gc
import uuid
import json
import time
import shutil
import zipfile
import logging
import argparse
import base64
import glob
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, NamedTuple, Tuple, Union, Optional
from platform import system
from netrc import netrc
import getpass
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Scientific & Data Libraries ---
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import geopandas as gpd
import h5py
import requests
import rioxarray
import rasterio
import folium
import pyproj
import shapely
import contextily as cx
from shapely import wkt
from shapely.geometry import box, Point, shape
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.transform import from_origin, Affine
from skimage.color import rgb2gray
from skimage import exposure

# --- Plotting ---
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# --- Widgets / UI ---
from ipywidgets import (Output, VBox, HBox, Layout, ToggleButton, Button, IntSlider, Play, jslink, Label, Checkbox)
from IPython.display import display, Javascript, Markdown

# --- Interactive Maps ---
from ipyleaflet import (Map, Marker, ImageOverlay, DrawControl, GeoJSON, FeatureGroup, LayersControl, CircleMarker, basemaps)

# --- Project-specific modules ---
import opera_utils
from opera_utils.disp._search import search

from disp_xr import (download, static_layers, product, stack as disp_stack, quality_metrics, io)
from mintpy.utils import utils as ut

# Decode byte arrays into actual strings
def decode_metadata_time(v):
    if isinstance(v, (np.ndarray, bytes)):
        return b"".join(v).decode("utf-8") if isinstance(v[0], (bytes, np.bytes_)) else str(v)
    return str(v)

def process_file(url, bbox, outdir, username, password):
    filename = url.split("/")[-1]
    base, ext = os.path.splitext(filename)
    outname = f"{outdir}/{base}.nc"

    if os.path.exists(outname):
        print(f"Skipped (exists): {filename}")
        return

    session = requests.Session()
    session.auth = (username, password)

    # Define compression settings
    comp = dict(zlib=True, complevel=4, shuffle=True)
    chunk_shape = (512, 512)

    try:
        with session.get(url, stream=True) as response:
            response.raise_for_status()
            file_bytes = BytesIO(response.content)

            try:
                with h5py.File(file_bytes, "r") as h5f:
                    def clip_bbox(ds, bbox):
                        height = ds.dims["y"]
                        width = ds.dims["x"]
                        x_start = max(0, min(width, bbox[0]))
                        x_stop = max(0, min(width, bbox[1]))
                        y_start = max(0, min(height, bbox[2]))
                        y_stop = max(0, min(height, bbox[3]))
                        if x_start >= x_stop or y_start >= y_stop:
                            return None
                        return (slice(y_start, y_stop), slice(x_start, x_stop))

                    if bbox is not None:
                        with xr.open_dataset(h5f, engine="h5netcdf") as ds:
                            clipped = clip_bbox(ds, bbox)
                            if clipped is None:
                                print(f"Skipped (bbox out of bounds): {filename}")
                                return
                            subset = ds.isel(y=clipped[0], x=clipped[1])
                            encoding = {
                                v: {**comp, "chunksizes": chunk_shape}
                                for v in subset.data_vars
                                if subset[v].dtype.kind in "fiu"
                            }
                            subset.to_netcdf(outname, engine="h5netcdf", encoding=encoding)

                        with xr.open_dataset(h5f, engine="h5netcdf", group="corrections") as ds_corr:
                            clipped = clip_bbox(ds_corr, bbox)
                            if clipped is not None:
                                corr_subset = ds_corr.isel(y=clipped[0], x=clipped[1])
                                encoding = {
                                    v: {**comp, "chunksizes": chunk_shape}
                                    for v in corr_subset.data_vars
                                    if corr_subset[v].dtype.kind in "fiu"
                                }
                                corr_subset.to_netcdf(outname, mode="a", group="corrections", engine="h5netcdf", encoding=encoding)
                    else:
                        with xr.open_dataset(h5f, engine="h5netcdf") as ds:
                            encoding = {
                                v: {**comp, "chunksizes": chunk_shape}
                                for v in ds.data_vars
                                if ds[v].dtype.kind in "fiu"
                            }
                            ds.to_netcdf(outname, engine="h5netcdf", encoding=encoding)

                        with xr.open_dataset(h5f, engine="h5netcdf", group="corrections") as ds_corr:
                            encoding = {
                                v: {**comp, "chunksizes": chunk_shape}
                                for v in ds_corr.data_vars
                                if ds_corr[v].dtype.kind in "fiu"
                            }
                            ds_corr.to_netcdf(outname, mode="a", group="corrections", engine="h5netcdf", encoding=encoding)

                    # Copy full metadata groups without subsetting or compression
                    for group in ["metadata", "identification"]:
                        try:
                            with h5py.File(outname, "a") as dest_hf:
                                if group in dest_hf:
                                    del dest_hf[group]
                                h5f.copy(group, dest_hf, name=group)
                        except Exception as e:
                            print(f"Failed to copy group '{group}': {e}")

                    copy_group_h5py(h5f, outname, "metadata/reference_orbit")
                    copy_group_h5py(h5f, outname, "metadata/secondary_orbit")

            finally:
                file_bytes.close()
                del file_bytes
                gc.collect()

    finally:
        session.close()

    print(f"Done: {filename}")

def extract_pixel_bbox_from_lalo(coord, lonlat_bbox):
    min_lon,  max_lon, min_lat, max_lat = lonlat_bbox.values()
    y0, x0 = coord.lalo2yx(min_lat, min_lon)
    y1, x1 = coord.lalo2yx(max_lat, max_lon)
    return slice(int(min(y0, y1)), int(max(y0, y1))), slice(int(min(x0, x1)), int(max(x0, x1)))

def copy_group_h5py(source_h5, target_path, group_name):
    try:
        with h5py.File(target_path, "a") as target_h5:
            src_group = source_h5[group_name]
            tgt_group = target_h5.require_group(group_name)

            for name, dataset in src_group.items():
                if name in tgt_group:
                    del tgt_group[name]
                tgt_ds = tgt_group.create_dataset(name, data=dataset[()])
                for key, val in dataset.attrs.items():
                    tgt_ds.attrs[key] = val

            for key, val in src_group.attrs.items():
                tgt_group.attrs[key] = val

    except Exception as e:
        print(f" Failed to copy {group_name} with h5py: {e}")

def get_metadata(disp_nc: str | Path | BytesIO| h5py.File, reference_date: Optional[str] = None) -> dict:
    """Get metadata for MINTPY from a DISP NetCDF file.

    Args:
        disp_nc (str or Path): The path to the DISP NetCDF file.
        reference_date (str, optional): The reference date. Defaults to None.

    Returns:
        dict: A dictionary containing the metadata.

    """
    # Get high-level metadata from DISP
    is_open_file = isinstance(disp_nc, h5py.File)
    if is_open_file:
        ds = disp_nc
    else:
        ds = h5py.File(disp_nc, "r")
    length, width = ds["displacement"][:].shape

    # Get general metadata
    metadata = {}
    for key, value in ds.attrs.items():
        metadata[key] = value

    for key, value in ds["identification"].items():
        value = value[()]
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        metadata[key] = value

    for key, value in ds["metadata"].items():
        # Skip unnecessary keys
        if key not in ["reference_orbit", "secondary_orbit", "processing_information"]:
            metadata[key] = value[()]

    metadata["x"] = ds["x"][:]
    metadata["y"] = ds["y"][:]
    metadata["length"] = length
    metadata["width"] = width
    ds.close()
    del ds

    # Get geospatial information
    geo_info = io.get_geospatial_info(disp_nc)

    ## Prepare it in mintpy atr format
    metadata["LENGTH"] = geo_info.rows
    metadata["WIDTH"] = geo_info.cols

    metadata["X_FIRST"] = geo_info.gt[0]
    metadata["Y_FIRST"] = geo_info.gt[3]
    metadata["X_STEP"] = geo_info.gt[1]
    metadata["Y_STEP"] = geo_info.gt[5]
    metadata["GT"] = geo_info.transform
    metadata["X_UNIT"] = metadata["Y_UNIT"] = "meters"
    metadata["WAVELENGTH"] = metadata["radar_wavelength"]
    metadata["REF_DATE"] = reference_date

    # Projection and UTM zone
    proj = CRS.from_wkt(geo_info.crs.wkt)
    epsg_code = proj.to_epsg()
    if str(epsg_code).startswith("326"):
        metadata["UTM_ZONE"] = str(epsg_code)[3:] + "N"
    elif str(epsg_code).startswith("327"):
        metadata["UTM_ZONE"] = str(epsg_code)[3:] + "S"
    else:
        metadata["UTM_ZONE"] = "UNKNOWN"
    metadata["EPSG"] = epsg_code

    # Hardcoded values
    metadata["ALOOKS"] = metadata["RLOOkS"] = 1
    metadata["EARTH_RADIUS"] = 6371000.0  # Hardcoded
    metadata["FILE_TYPE"] = "timeseries"
    metadata["UNIT"] = "m"
    metadata["AZIMUTH_PIXEL_SIZE"] = 14.1  # where this comes from

    # Datetime
    try:
      t = pd.to_datetime([metadata["reference_zero_doppler_start_time"], metadata["reference_zero_doppler_end_time"],])
    except:
      start_time = decode_metadata_time(metadata["reference_zero_doppler_start_time"])
      end_time = decode_metadata_time(metadata["reference_zero_doppler_end_time"])

      # Convert to datetime
      t = pd.to_datetime([start_time, end_time])
    t_mid = t[0] + t.diff()[1] / 2
    total_seconds = (t_mid.hour * 3600 + t_mid.minute * 60 + t_mid.second + t_mid.microsecond / 1e6)
    metadata["CENTER_LINE_UTC"] = total_seconds

    # Clean up of metadata dicts
    try:
      for key in ["reference_datetime", "secondary_datetime"]:
         del metadata[key]
    except:
      pass
    return metadata

def display_opera_frames_map(geojson_path="Frames_Information.geojson"):
    """Download and display OPERA DISP-S1 frames on a Folium map."""
    if not os.path.exists(geojson_path):
        os.system(f"wget -q https://raw.githubusercontent.com/OPERA-Cal-Val/OPERA_Applications/main/DISP/Discover/Frames_Information.geojson")

    gdf = gpd.read_file(geojson_path)
    m = folium.Map(location=[30, -100], zoom_start=4, tiles='Esri.WorldImagery')

    popup_fields = ["frame_id", "orbit_pass", "num_dates"]

    # Ascending
    asc = gdf[gdf.orbit_pass == "ASCENDING"]
    asc_layer = folium.FeatureGroup(name="Ascending", show=True)
    folium.GeoJson(asc,name="Ascending",style_function=lambda x: {"color": "blue"},tooltip=folium.GeoJsonTooltip(fields=popup_fields, aliases=["Frame ID", "Pass", "# Dates"])).add_to(asc_layer)

    # Descending
    desc = gdf[gdf.orbit_pass == "DESCENDING"]
    desc_layer = folium.FeatureGroup(name="Descending", show=True)
    folium.GeoJson(desc,name="Descending",style_function=lambda x: {"color": "red"},tooltip=folium.GeoJsonTooltip(fields=popup_fields, aliases=["Frame ID", "Pass", "# Dates"])).add_to(desc_layer)

    asc_layer.add_to(m)
    desc_layer.add_to(m)
    folium.LayerControl().add_to(m)

    return m

def setup_earthdata_credentials(netrc_path="/root/.netrc"):
    """Prompt user for NASA Earthdata credentials and create .netrc file."""
    username = input("NASA Earthdata username: ")
    password = getpass.getpass("NASA Earthdata password: ")

    netrc = Path(netrc_path)
    netrc.write_text(f"machine urs.earthdata.nasa.gov\nlogin {username}\npassword {password}\n")
    netrc.chmod(0o600)

    print(f"File {netrc_path} created.")

def prompt_user_and_search():
    """Prompt for frame/date input, search ASF DAAC, create folders, and estimate stack size."""
    frame_id = int(input("Enter the frame ID (e.g., 34478): "))
    start_str = input("Enter the start date (YYYY-MM-DD): ")
    end_str = input("Enter the end date (YYYY-MM-DD): ")

    # Parse dates
    start_datetime = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_datetime = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    print(f"Using Frame {frame_id}, from {start_datetime.date()} to {end_datetime.date()}")

    # Setup directories
    work_dir = Path(f"{frame_id}").resolve()
    nc_dir = work_dir / "OPERA_DISP_S1_Files"
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(nc_dir, exist_ok=True)

    # Get Earthdata credentials and Determine the correct netrc file path across platforms
    netrc_file = "_netrc" if os.name == "nt" else ".netrc"
    netrc_path = Path(os.path.expanduser(f"~/{netrc_file}"))
    username, _, password = netrc(str(netrc_path)).authenticators("urs.earthdata.nasa.gov")

    # Search ASF DAAC
    products_df = download.search(frame_id=frame_id, start_datetime=start_datetime, end_datetime=end_datetime)

    # Get URLs
    nc_urls = [line.strip() for line in products_df['filename'].values if isinstance(line, str) and line.strip()]
    print(f"Number of DISP F{frame_id} granules: {products_df.shape[0]}")

    # Estimate size
    session = requests.Session()
    session.auth = (username, password)
    response = session.get(nc_urls[0])
    response.raise_for_status()
    file_bytes = BytesIO(response.content)
    file_size = file_bytes.getbuffer().nbytes / 1024**2
    est_gb = (len(nc_urls) * file_size) / 1024
    print(f"File size: {file_size:.2f} MB")
    print(f"This will result in a stack of {est_gb:.2f} Gb, without cropping")
    print("Please make sure you have enough space or consider cropping the stack")

    return products_df, frame_id, start_datetime, end_datetime, work_dir, nc_dir, nc_urls, username, password

def setup_interactive_bbox_map(products_df):
    """Creates an interactive map with OPERA DISP footprints and draw tool to define a bounding box."""
    draw_out = Output()
    point_out = Output()

    # Prepare GeoDataFrame
    gdf = gpd.GeoDataFrame(products_df.copy(), crs=4326)
    gdf = gdf.map(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)

    # Get map center
    center = gdf.geometry.union_all().centroid
    m = Map(center=(center.y, center.x), zoom=6, scroll_wheel_zoom=True)

    # GeoJSON layer
    style = {'color': 'blue','weight': 1,'fillColor': 'transparent','fillOpacity': 0.0}
    hover_style = {'color': 'blue','fillOpacity': 0.1}

    geojson_layer = GeoJSON(data=gdf.__geo_interface__,name="OPERA_DISP",style=style,hover_style=hover_style)
    m.add_layer(geojson_layer)

    # Draw control
    draw_control = DrawControl(rectangle={"shapeOptions": {"fillColor": "#6be","color": "red","fillOpacity": 0.0,"weight": 2}}, polygon={}, polyline={}, circle={}, circlemarker={})

    # Global bounding box dictionary to hold output
    bbox_bounds = {}

    @draw_control.on_draw
    def handle_draw(target, action, geo_json):
        geom = shape(geo_json['geometry'])
        bounds = geom.bounds
        bbox_bounds.clear()
        bbox_bounds.update({"lon_min": bounds[0],"lon_max": bounds[2],"lat_min": bounds[1],"lat_max": bounds[3]})
        with draw_out:
            draw_out.clear_output()
            print("Bounding Box (lon/lat):")
            print(bbox_bounds)

    m.add_control(draw_control)
    display(VBox([m, draw_out, point_out]))
    
    return bbox_bounds

def estimate_stack_size(nc_url, bbox_bounds, username, password):
    """Estimate the subset and stack size for a DISP file.

    Args:
        nc_url (str): URL to the first DISP NetCDF file.
        bbox_bounds (dict): Bounding box with lat/lon keys: lon_min, lon_max, lat_min, lat_max.
        username (str): Earthdata username.
        password (str): Earthdata password.

    Returns:
        tuple: (file_size_MB, subset_MB, ratio_percent, total_stack_size_GB, pixel_bbox)
    """
    session = requests.Session()
    session.auth = (username, password)
    response = session.get(nc_url[0])
    response.raise_for_status()

    file_bytes = BytesIO(response.content)
    file_size = file_bytes.getbuffer().nbytes / 1024**2  # in MB

    # Extract metadata
    meta = get_metadata(file_bytes)
    coord = ut.coordinate(meta)
    y_slice, x_slice = extract_pixel_bbox_from_lalo(coord, bbox_bounds)

    row_start, row_end = y_slice.start, y_slice.stop
    col_start, col_end = x_slice.start, x_slice.stop
    pixel_bbox = [col_start, col_end, row_start, row_end]

    # Estimate subset size
    with xr.open_dataset(file_bytes, engine="h5netcdf") as ds:
        full_height, full_width = ds.dims["y"], ds.dims["x"]

        # Clip pixel bounds safely
        col_start = max(0, min(full_width, col_start))
        col_end = max(0, min(full_width, col_end))
        row_start = max(0, min(full_height, row_start))
        row_end = max(0, min(full_height, row_end))

        subset_height = row_end - row_start
        subset_width = col_end - col_start

        if subset_height == 0 or subset_width == 0:
            ratio = 0
            subset_file_size = 0
        else:
            ratio = (subset_height * subset_width) / (full_height * full_width)
            raw_nbytes = ds.nbytes
            actual_compression_ratio = (file_size * 1024**2) / raw_nbytes
            subset_file_size = (raw_nbytes * ratio * actual_compression_ratio) / 1024**2  # MB

    return file_size, subset_file_size, ratio, (subset_file_size * len(nc_url)) / 1024, pixel_bbox

def download_disp_files(nc_urls, bbox, outdir, username, password, num_workers=3):
    """
    Download and optionally crop DISP NetCDF files in parallel.

    Args:
        nc_urls (list): List of DISP file URLs.
        bbox (list or None): Pixel bounding box [col_start, col_end, row_start, row_end] or None for full frame.
        outdir (str or Path): Output directory for NetCDF files.
        username (str): Earthdata username.
        password (str): Earthdata password.
        num_workers (int): Number of parallel threads.
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_url = {executor.submit(process_file, url, bbox, outdir, username, password): url for url in nc_urls}
        for future in as_completed(future_to_url):
            _ = future.result()

def extract_static_layers(disp_file, output_dir="static"):
    """
    Extract static layers (e.g., water mask, landcover, DEM) from a DISP file.

    Args:
        disp_file (str or Path): Path to a DISP NetCDF file.
        output_dir (str): Directory where the static layers will be stored.
    """
    static_layers.get_static_layers(disp_file, output_dir=output_dir)

def download_and_plot_dem(disp_file, output_dir="static", show=True):
    """
    Download DEM for the DISP file and optionally display it.

    Args:
        disp_file (str or Path): Path to DISP NetCDF file.
        output_dir (str): Output directory for the DEM.
        show (bool): Whether to display the DEM with matplotlib.
    
    Returns:
        str: Path to the downloaded DEM file.
    """
    logging.getLogger("rasterio").setLevel(logging.ERROR)

    dem_path = static_layers.download_dem(disp_file, output_dir=output_dir)

    if show:
        img = io.open_image(dem_path)[0]
        plt.imshow(img)
        plt.colorbar()
        plt.title("Downloaded DEM")
        plt.show()

    return dem_path

def setup_workspace_and_metadata(frame_id: int, base_path: str = ".", verbose: bool = True):
    """
    Set up working directory for a given frame ID and extract metadata from the first DISP NetCDF.

    Args:
        frame_id (int): Frame ID number (e.g., 24455).
        base_path (str): Base directory path where frame folder will be created.
        verbose (bool): Whether to print paths and status info.

    Returns:
        tuple: (Path to working directory, Path to nc_dir, list of nc_files, metadata dict)
    """

    work_dir = Path(base_path) / str(frame_id)
    nc_dir = work_dir / "OPERA_DISP_S1_Files"
    os.makedirs(nc_dir, exist_ok=True)

    nc_files = list(nc_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {nc_dir}")

    metadata = get_metadata(nc_files[0])

    if verbose:
        print(f"Workspace: {work_dir}")
        print(f"Found {len(nc_files)} NetCDF files.")

    return work_dir, nc_dir, nc_files, metadata

def get_disp_versions(nc_dir: Path):
    """
    Loads DISP NetCDFs and groups them by version.

    Args:
        nc_dir (Path): Directory containing DISP NetCDF files.

    Returns:
        dict: Dictionary of version -> filtered DataFrame
    """
    disp_df = product.get_disp_info(nc_dir)
    versions = {}
    for v in disp_df.version.unique():
        versions[v] = disp_df[disp_df.version == v].copy()
        print(f"{v}, size: {versions[v].shape[0]}")
    return disp_df, versions

def plot_date_scatter_by_version(versions_dict: dict):
    """
    Plots Date1 vs. Date2 scatter plots for each DISP version.

    Args:
        versions_dict (dict): Output from get_disp_versions()
    """
    for version, df in versions_dict.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_date1_date2(ax, df)
        ax.set_title(f'Version: {version}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_date1_date2(ax, df):
    """Helper function to scatter Date1 vs Date2 on a matplotlib axis."""
    df = df.copy()
    df["date1"] = pd.to_datetime(df["date1"], format="%Y%m%d")
    df["date2"] = pd.to_datetime(df["date2"], format="%Y%m%d")
    ax.scatter(df.date1, df.date2, alpha=0.7, marker="o", color="b")
    ax.set_xlabel("Date1 (Start Date)")
    ax.set_ylabel("Date2 (End Date)")
    ax.set_title("Date1 vs. Date2")
    ax.grid(True)

def build_stack_and_get_epsg(disp_df):
    """
    Combine displacement products into a stack and extract EPSG code.

    Args:
        disp_df (DataFrame): DataFrame returned from product.get_disp_info()

    Returns:
        stack_prod (xarray.Dataset): Combined stack dataset
        epsg (int): EPSG code of the spatial reference
    """
    stack_prod = disp_stack.combine_disp_product(disp_df)
    epsg = pyproj.CRS(stack_prod.spatial_ref.attrs['crs_wkt']).to_epsg()
    return stack_prod, epsg

def create_reference_selection_map(stack_prod, meta):

    print("Please enable clicking point and choose a reference point by clicking on the map")

    # Prepare image from xarray
    da = stack_prod.isel(time=-1).displacement
    mask = stack_prod.isel(time=-1).recommended_mask
    water = stack_prod.isel(time=-1).water_mask
    combined_mask = (mask != 1) | (water != 1)
    da = da.where(~combined_mask)

    da.rio.write_crs(meta["EPSG"], inplace=True)
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    da_3857 = da.rio.reproject("EPSG:3857", resampling=Resampling.bilinear)

    vmin, vmax = np.nanpercentile(da_3857.values, [2, 98])
    norm_img = np.clip((da_3857.values - vmin) / (vmax - vmin), 0, 1)
    norm_img[da_3857.values == 0] = np.nan

    cmap = cm.get_cmap('RdBu_r').copy()
    cmap.set_bad(color=(0, 0, 0, 0))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.axis("off")
    ax.imshow(norm_img, cmap=cmap, origin='upper')
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)

    west, south, east, north = da_3857.rio.bounds()
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon_west, lat_south = transformer.transform(west, south)
    lon_east, lat_north = transformer.transform(east, north)
    center_lon, center_lat = transformer.transform((west + east) / 2, (south + north) / 2)

    m = Map(center=(center_lat, center_lon),zoom=12,scroll_wheel_zoom=True,basemap=basemaps.Esri.WorldImagery,layout=Layout(width="100%", height="700px"))

    overlay = ImageOverlay(url='data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8'),bounds=((lat_south, lon_west), (lat_north, lon_east)),opacity=0.6)
    m.add_layer(overlay)

    draw_out = Output()
    click_toggle = ToggleButton(value=False, description='Enable Point Click', button_style='info')
    clicked_ref_group = FeatureGroup(name='Selected Point')
    m.add_layer(clicked_ref_group)

    clicked_ref = {}
    clicked_refe = []

    def handle_map_click(**kwargs):
        nonlocal clicked_ref
        if click_toggle.value and kwargs.get('type') == 'click':
            lat, lon = kwargs['coordinates']
            clicked_ref = {"lat": lat, "lon": lon}
            clicked_refe.append(clicked_ref)
            clicked_ref_group.clear_layers()
            clicked_ref_group.add_layer(Marker(location=(lat, lon)))
            with draw_out:
                draw_out.clear_output()
                print("Clicked Point (lat/lon):")
                print(clicked_ref)

    m.on_interaction(handle_map_click)
    m.add_control(LayersControl())

    display(VBox([click_toggle, m, draw_out]))

    return clicked_refe  # Will only update after interaction

def set_reference_point(stack_prod, clicked_ref):
    """
    Re-references the displacement stack to a selected point.

    Args:
        stack_prod (xr.Dataset): The full displacement stack.
        clicked_ref (dict): Dictionary with keys 'lat' and 'lon'.

    Returns:
        xr.Dataset: Stack with displacement referenced to the selected point.
    """
    
    reference_yx = (clicked_ref['lat'], clicked_ref['lon'])
    epsg = pyproj.CRS(stack_prod.spatial_ref.attrs['crs_wkt']).to_epsg()

    ref_x_utm, ref_y_utm = latlon_to_utm(*reference_yx, epsg)
    ref_disp = stack_prod.sel(y=ref_y_utm, x=ref_x_utm, method='nearest').displacement

    stack_prod['displacement'] = stack_prod.displacement - ref_disp

    return stack_prod

def plot_displacement_map(stack_prod, meta, clicked_ref, buffer_fraction=0.12, velocity=None):
    """
    Plots cumulative displacement or velocity map over a basemap with a reference point.

    Args:
        stack_prod (xr.Dataset): The stacked displacement dataset.
        meta (dict): Metadata including CRS (e.g., EPSG code).
        clicked_ref (dict): Dictionary with 'lat' and 'lon'.
        buffer_fraction (float): Fractional padding around data extent.
        velocity_da (xr.DataArray, optional): Optional velocity DataArray to plot instead of displacement.

    Returns:
        None
    """

    def get_baseimage_from_bounds(bounds, source, grayscale=False, gamma=None, log=None, zoom='auto'):
        xmin, ymin, xmax, ymax = bounds
        left, right, bottom, top = cx.plotting._reproj_bb(xmin, xmax, ymin, ymax, 'EPSG:4326', "epsg:3857")
        image, extent = cx.tile.bounds2img(left, bottom, right, top, zoom=zoom, source=source, ll=False)
        image, extent = cx.tile.warp_tiles(image, extent, t_crs='EPSG:4326', resampling=Resampling.bilinear)
        if grayscale:
            image = rgb2gray(image[:, :, :3])
        if gamma is not None:
            image = exposure.adjust_gamma(image, gamma)
        if log is not None:
            image = exposure.adjust_log(image, log)
        return image, extent

    # --- Determine data to plot ---
    if velocity is not None:
        da = velocity.velocity * 100  # cm/year
        is_velocity = True
    else:
        if "displacement" not in stack_prod:
            raise ValueError("Dataset does not contain 'displacement' variable.")
        da = stack_prod.isel(time=-1).displacement * 100  # cm cumulative
        is_velocity = False

    mask = stack_prod.isel(time=-1).recommended_mask
    water = stack_prod.isel(time=-1).water_mask
    da = da.where((mask == 1) & (water == 1))

    # Set CRS and reproject
    da.rio.write_crs(meta["EPSG"], inplace=True)
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    da_4326 = da.rio.reproject("EPSG:4326", resampling=Resampling.bilinear)

    # Buffer bounds
    xmin, ymin, xmax, ymax = da_4326.rio.bounds()
    x_pad = (xmax - xmin) * buffer_fraction
    y_pad = (ymax - ymin) * buffer_fraction
    buffered_bounds = [xmin - x_pad, ymin - y_pad, xmax + x_pad, ymax + y_pad]
    extent = [buffered_bounds[0], buffered_bounds[2], buffered_bounds[1], buffered_bounds[3]]

    # Normalize
    v = np.nanpercentile(da_4326.values, [2, 98])
    vmax = max(abs(v[0]), abs(v[1]))
    vmin = -vmax

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Add basemap
    bg, bg_extent = get_baseimage_from_bounds(buffered_bounds,source=cx.providers.Esri.WorldImagery,grayscale=True,gamma=0.9)
    ax.imshow(bg, extent=bg_extent, cmap="gray", origin="upper", zorder=1)

    # Displacement overlay
    ax.imshow(da_4326.values,extent=[xmin, xmax, ymin, ymax],cmap=cm.get_cmap("RdBu_r"),vmin=vmin,vmax=vmax,alpha=0.6,origin="upper",zorder=2)

    # Reference point
    ref_pt = gpd.GeoDataFrame(geometry=[Point(clicked_ref["lon"], clicked_ref["lat"])],crs="EPSG:4326")
    ref_pt.plot(ax=ax, color="green", markersize=30, label="Reference Point", zorder=3)

    # Colorbar
    cbar = plt.colorbar(ax.images[1], ax=ax, shrink=0.6, pad=0.04)
    if is_velocity:
        cbar.set_label("Velocity [cm/year]")
    else:
        cbar.set_label("Displacement [cm]")

    # Add ticks
    ax.set_xticks(np.linspace(extent[0], extent[1], 5))
    ax.set_yticks(np.linspace(extent[2], extent[3], 5))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}°E"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}°N"))
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # Attribution
    ax.annotate(
        "Tiles © Esri - Sources: Esri, DigitalGlobe, GeoEye, i-cubed, USDA FSA, USGS, AEX, "
        "Getmapping, Aerogrid, IGN, IGP, swisstopo, and the GIS User Community",
        xy=(0.5, -0.12), xycoords='axes fraction',
        ha='center', va='top', fontsize=6, color='gray'
    )

    ax.legend()
    title = "Velocity Map (cm/year)" if is_velocity else "Cumulative Displacement (cm)"
    ax.set_title(f"{title}")
    plt.tight_layout()
    plt.show()

def latlon_to_utm(lat, lon, epsg):
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    return transformer.transform(lon, lat)

def create_image_overlay(filepath, bounds, name="overlay", opacity=0.6):
    from PIL import Image
    from base64 import b64encode
    from io import BytesIO
    import uuid

    ext = os.path.splitext(filepath)[1][1:]
    image = Image.open(filepath)

    buf = BytesIO()
    image.save(buf, format=ext.upper())
    base64_str = b64encode(buf.getvalue()).decode('ascii')

    # Force a unique overlay URL to prevent caching
    url = f"data:image/{ext};base64,{base64_str}#v={uuid.uuid4()}"

    return ImageOverlay(url=url, bounds=bounds, name=name, opacity=opacity)

def setup_interactive_viewer(stack_prod):
    from IPython.display import Javascript
    display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))

    # Globals
    previous_overlay_file = None
    current_overlay = None
    clicked_ref = None
    clicked_plot = None

    # Generate overlay image
    def generate_overlay_image(stack_prod, clicked_ref=None, vmin=None, vmax=None, time_index=-1):
        da = stack_prod.displacement

        if clicked_ref is not None:
            epsg = pyproj.CRS(stack_prod.spatial_ref.attrs["crs_wkt"]).to_epsg()
            ref_x, ref_y = latlon_to_utm(clicked_ref["lat"], clicked_ref["lon"], epsg)
            ref_disp = stack_prod.sel(x=ref_x, y=ref_y, method="nearest").displacement
            da = da - ref_disp

        da = da.isel(time=time_index)
        mask = stack_prod.isel(time=time_index).recommended_mask
        water = stack_prod.isel(time=time_index).water_mask
        da = da.where((mask == 1) & (water == 1))
        da = da * 100

        crs_wkt = stack_prod.spatial_ref.attrs["crs_wkt"]
        da.rio.write_crs(crs_wkt, inplace=True)
        da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        da_3857 = da.rio.reproject("EPSG:3857", resampling=Resampling.bilinear)

        west, south, east, north = da_3857.rio.bounds()
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon_west, lat_south = transformer.transform(west, south)
        lon_east, lat_north = transformer.transform(east, north)
        center_lon, center_lat = transformer.transform((west + east) / 2, (south + north) / 2)

        cmap = cm.get_cmap("RdBu_r").copy()
        cmap.set_bad((0, 0, 0, 0))
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.axis("off")
        ax.imshow(da_3857.values, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)

        if clicked_ref:
            ref_x_3857, ref_y_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(clicked_ref["lon"], clicked_ref["lat"])
            ix = int(np.argmin(np.abs(da_3857.x.values - ref_x_3857)))
            iy = int(np.argmin(np.abs(da_3857.y.values - ref_y_3857)))
            ax.plot(ix, iy, 'o', color='red', markeredgecolor='black')

        filename = f"overlay_{uuid.uuid4().hex}.png"
        fig.savefig(filename, bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)

        overlay = create_image_overlay(filepath=filename,bounds=((lat_south, lon_west), (lat_north, lon_east)),name=f"Overlay",opacity=0.6,)
        return overlay, (center_lat, center_lon), filename

    # Initialize widgets
    draw_out = Output()
    plot_out = Output()
    click_toggle = ToggleButton(value=False, description="Enable Reference Click", button_style="info")
    plot_toggle = ToggleButton(value=False, description="Enable Plot Click", button_style="warning")
    clicked_ref_group = FeatureGroup(name="Reference")
    clicked_plot_group = FeatureGroup(name="Plot Point")

    # Mode label to display current action
    mode_label = Label(value="Current Mode: None")

    def on_toggle(change):
        if change["owner"] == click_toggle and change["new"]:
            plot_toggle.value = False
            mode_label.value = "Current Mode: Set Reference Point"
        elif change["owner"] == plot_toggle and change["new"]:
            click_toggle.value = False
            mode_label.value = "Current Mode: Select Plot Point"
        elif not click_toggle.value and not plot_toggle.value:
            mode_label.value = "Current Mode: None"

    click_toggle.observe(on_toggle, names="value")
    plot_toggle.observe(on_toggle, names="value")

    # Prepare map and data
    vals = stack_prod.isel(time=-1).displacement.values * 100
    absmax = max(np.abs(np.nanpercentile(vals, [2, 98])))
    global_vmin, global_vmax = -absmax, absmax

    initial_overlay, (center_lat, center_lon), filename = generate_overlay_image(stack_prod, vmin=global_vmin, vmax=global_vmax)

    m = Map(center=(center_lat, center_lon), zoom=12, scroll_wheel_zoom=True, basemap=basemaps.Esri.WorldImagery, layout=Layout(height="700px", width="100%"))
    m.add_layer(initial_overlay)
    m.add_layer(clicked_ref_group)
    m.add_layer(clicked_plot_group)
    m.add_control(LayersControl())
    current_overlay = initial_overlay
    previous_overlay_file = filename

    # Time slider
    time_slider = IntSlider(value=stack_prod.sizes["time"] - 1, min=0, max=stack_prod.sizes["time"] - 1, description="Time")
    play = Play(interval=500, value=time_slider.value, min=time_slider.min, max=time_slider.max)
    jslink((play, 'value'), (time_slider, 'value'))
    date_label = Label()

    def on_time_change(change):
        nonlocal current_overlay, previous_overlay_file, clicked_ref

        t = change["new"]
        date_label.value = str(pd.to_datetime(stack_prod.time.values[t]).date())

        overlay, _, filename = generate_overlay_image(stack_prod, clicked_ref, global_vmin, global_vmax, t)
        if current_overlay in m.layers:
            m.substitute_layer(current_overlay, overlay)
        else:
            m.add_layer(overlay)

        try:
            if previous_overlay_file and os.path.exists(previous_overlay_file):
                os.remove(previous_overlay_file)
        except Exception as e:
            print(f"Failed to delete old overlay: {e}")

        current_overlay = overlay
        previous_overlay_file = filename

    time_slider.observe(on_time_change, names="value")
    on_time_change({'new': time_slider.value})

    # Click handler
    def handle_map_click(**kwargs):
        nonlocal clicked_ref, clicked_plot, current_overlay, previous_overlay_file

        if kwargs.get("type") != "click":
            return
        lat, lon = kwargs["coordinates"]

        if click_toggle.value:
            clicked_ref = {"lat": lat, "lon": lon}
            clicked_ref_group.clear_layers()
            clicked_ref_group.add_layer(Marker(location=(lat, lon)))
            draw_out.clear_output()
            with draw_out:
                print(f"Reference set to lat={lat:.6f}, lon={lon:.6f}")
            on_time_change({'new': time_slider.value})

        elif plot_toggle.value:
            clicked_plot = {"lat": lat, "lon": lon}
            clicked_plot_group.clear_layers()
            clicked_plot_group.add_layer(Marker(location=(lat, lon), color='blue'))
            draw_out.clear_output()
            with draw_out:
                print(f"Plot point set to lat={lat:.6f}, lon={lon:.6f}")

    m.on_interaction(handle_map_click)

    # Plot button
    plot_button = Button(description="Plot Time Series", button_style="primary")

    def plot_time_series(_):
        plot_out.clear_output()

        if not clicked_plot:
            with plot_out:
                print("Select a point with plot mode.")
            return

        epsg = pyproj.CRS.from_wkt(stack_prod.spatial_ref.attrs["crs_wkt"]).to_epsg()
        x_click, y_click = latlon_to_utm(clicked_plot["lat"], clicked_plot["lon"], epsg)

        # Find nearest grid point
        x_coords = stack_prod.x.values
        y_coords = stack_prod.y.values
        x_nearest = x_coords[np.argmin(np.abs(x_coords - x_click))]
        y_nearest = y_coords[np.argmin(np.abs(y_coords - y_click))]

        # Compute Euclidean distance in meters
        distance = np.sqrt((x_click - x_nearest)**2 + (y_click - y_nearest)**2)

        # Set distance threshold (meters)
        threshold_m = 100
        if distance > threshold_m:
            with plot_out:
                print(f"❌ Point too far from valid pixel (distance = {distance:.1f} m).")
                print("Please click closer to a valid area.")
            return

        # Check masks (from the last time index - or choose a specific time)
        mask_val = stack_prod.sel(x=x_nearest, y=y_nearest, method="nearest").isel(time=-1).recommended_mask.values
        water_val = stack_prod.sel(x=x_nearest, y=y_nearest, method="nearest").isel(time=-1).water_mask.values

        if mask_val != 1 or water_val != 1:
            with plot_out:
                print("❌ Selected point is masked out or located on water.")
                print("Please choose a valid terrestrial point.")
            return

        # Get displacement
        disp = stack_prod.sel(x=x_nearest, y=y_nearest, method="nearest").displacement

        # Apply reference if available
        if clicked_ref:
            ref_x, ref_y = latlon_to_utm(clicked_ref["lat"], clicked_ref["lon"], epsg)
            ref_disp = stack_prod.sel(x=ref_x, y=ref_y, method="nearest").displacement
            disp -= ref_disp

        # Plot
        with plot_out:
            fig, ax = plt.subplots(figsize=(10, 4))
            dates = pd.to_datetime(stack_prod.time.values)
            ax.scatter(dates, disp.values * 100, marker='o', color='black', s=10)
            ax.set_title(f"Displacement time series (cm)\nLat={clicked_plot['lat']:.4f}, Lon={clicked_plot['lon']:.4f}")
            # Axis labels
            ax.set_xlabel("Time")
            ax.set_ylabel("Displacement (cm)")

            # Date formatting
            import matplotlib.dates as mdates
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            #fig.autofmt_xdate(rotation=0)

            # Optional: adjust layout
            plt.tight_layout()

            plt.grid(True)

            plt.show()

    plot_button.on_click(plot_time_series)

    # Display UI
    display(VBox([
        HBox([click_toggle, plot_toggle, plot_button]),
        mode_label,
        HBox([time_slider, date_label]),
        m,
        draw_out,
        plot_out
    ]))

def extract_ministack_references(stack_prod):
    """Extract quality layers and unique reference-date ministacks."""

    stack_quality = stack_prod.chunk({'time': -1})
    ref_dates, indices = np.unique(stack_quality.reference_time, return_index=True)
    ministack_ref = stack_prod.isel(time=indices).chunk({'time': -1})

    return stack_quality, ministack_ref

def print_stackprod_variable_info(stack_prod):
    """Print variable names and their attributes in an xarray Dataset."""
    variables = list(stack_prod.data_vars.items())

    output_lines = []

    for var_name, da in variables[3:]:
        output_lines.append(f"###  Layer: `{var_name}`")
        for attr_name, attr_val in da.attrs.items():
            output_lines.append(f"- **{attr_name}**: {attr_val}")
        output_lines.append("")  # empty line between variables

    # Display once, in one cell
    display(Markdown("\n".join(output_lines)))

def plot_quality_summary(ministack_ref, stack_quality):
    """
    Compute and plot quality metrics:
    - % Persistent Scatterers (PS)
    - % Valid Timeseries Pixels (recommended_mask)
    - % Connected Component valid (not 0)
    """

    # Check for minimum time steps
    if ministack_ref.persistent_scatterer_mask.sizes["time"] <= 2:
        print("Not enough Ministacks steps to compute quality summary. At least 2 required.")
        return

    # Compute metrics
    pct_ps = quality_metrics.get_value_percentage(ministack_ref.persistent_scatterer_mask.isel(time=slice(1, None)), value=1)
    pct_mask = quality_metrics.get_value_percentage(ministack_ref.recommended_mask.isel(time=slice(1, None)), value=1, reverse=False)
    pct_conncomp = quality_metrics.get_value_percentage(stack_quality.connected_component_labels.isel(time=slice(1, None)), value=0, reverse=True)

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    im1 = ax[0].imshow(ma.masked_equal(pct_ps, 0), cmap='bone_r', clim=[0, 30], interpolation='nearest')
    im2 = ax[1].imshow(ma.masked_equal(pct_mask, 0), cmap='plasma', clim=[0, 100], interpolation='nearest')
    im3 = ax[2].imshow(ma.masked_equal(pct_conncomp, 0), cmap='afmhot_r', clim=[0, 100], interpolation='nearest')

    for a, im in zip(ax, [im1, im2, im3]):
        fig.colorbar(im, ax=a, location='bottom')

    for a, title in zip(ax, ['Pct of PS in stack','Timeseries density','Pct of valid conncomp']):
        a.set_title(title)

    plt.tight_layout()
    plt.show()

def plot_advanced_quality_metrics(stack_quality):
    """
    Compute and plot advanced quality metrics:
    - Median temporal coherence
    - Median phase similarity
    - Total number of 2π phase jumps
    """

    # Compute statistics
    median_tcoh = quality_metrics.get_stack_stat(stack_quality.temporal_coherence.isel(time=slice(1, None)), mode='median')
    median_psim = quality_metrics.get_stack_stat(stack_quality.phase_similarity.isel(time=slice(1, None)), mode='median')
    inv_res_sum = quality_metrics.get_stack_stat(stack_quality.timeseries_inversion_residuals.isel(time=slice(1, None)), mode='sum')
    num_2pi_jump = inv_res_sum / (2 * np.pi)

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    im1 = ax[0].imshow(ma.masked_equal(median_tcoh, 0), cmap='afmhot', clim=[0, 1], interpolation='nearest')
    im2 = ax[1].imshow(ma.masked_equal(median_psim, 0), cmap='afmhot', clim=[0, 1], interpolation='nearest')
    im3 = ax[2].imshow(ma.masked_equal(num_2pi_jump, 0), cmap='plasma', clim=[0, 300], interpolation='nearest')

    for a, im in zip(ax, [im1, im2, im3]):
        fig.colorbar(im, ax=a, location='bottom')

    for a, title in zip(ax, ['Median temporal coherence','Median Phase similarity','N of 2π jumps']):
        a.set_title(title)

    plt.tight_layout()
    plt.show()

def plot_shp_stats(ministack_ref):
    """
    Compute and plot median and standard deviation of shp_counts from a ministack.

    Parameters:
        ministack_ref (xarray.Dataset): Subset of the displacement stack, reference dates only.
    """
    # Check for enough time steps
    if ministack_ref.shp_counts.sizes.get("time", 0) <= 2:
        print("Not enough time steps to compute shp stats. At least 2 required.")
        return

    # Compute statistics
    shp_median = quality_metrics.get_stack_stat(ministack_ref.shp_counts.isel(time=slice(1, None)), mode='median')
    shp_std = quality_metrics.get_stack_stat(ministack_ref.shp_counts.isel(time=slice(1, None)), mode='std')

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax[0].imshow(ma.masked_equal(shp_median, 0), cmap='bone_r', interpolation='nearest')
    im2 = ax[1].imshow(ma.masked_equal(shp_std, 0), cmap='afmhot_r', interpolation='nearest')

    for a, im in zip(ax, [im1, im2]):
        fig.colorbar(im, ax=a, location='bottom')

    for a, txt in zip(ax, ['Shp median', 'Shp std']):
        a.set_title(txt)

    plt.tight_layout()
    plt.show()

def compute_velocity_from_stack(stack_prod, displacement_var="displacement"):
    """
    Compute linear velocity from a displacement time series in an xarray dataset.

    Parameters:
    ----------
    stack_prod : xarray.Dataset
        Dataset containing a 3D displacement variable with dimensions (time, y, x).
    displacement_var : str
        Name of the displacement variable in the dataset.

    Returns:
    -------
    velocity_ds : xarray.Dataset
        Dataset containing the computed velocity as a DataArray (units: m/year).
    """

    # Extract displacement and time
    disp = stack_prod[displacement_var].values  # shape: (nt, ny, nx)
    times = stack_prod['time'].values           # datetime64 array

    nt, ny, nx = disp.shape

    # Convert time to decimal years
    def _decimal_year(dates):
        dates = pd.to_datetime(dates)
        return dates.year + (dates.dayofyear - 1) / 365.25

    tdecimal = _decimal_year(times)  # shape (nt,)
    A = np.vstack([tdecimal, np.ones_like(tdecimal)]).T  # shape: (nt, 2)

    # Reshape displacement to (nt, ny*nx)
    y = disp.reshape(nt, -1)

    # Solve least squares for linear fit: displacement = velocity * time + intercept
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)  # shape: (2, ny*nx)
    velocity = coef[0].reshape(ny, nx).astype(np.float32)

    # Create velocity DataArray
    vel_da = xr.DataArray(velocity,dims=("y", "x"),coords={"y": stack_prod.y, "x": stack_prod.x},
        attrs={
            "long_name": "Velocity",
            "units": "m/year",
            "description": "Linear velocity estimated from displacement time series",
            "start_date": str(times[0]),
            "end_date": str(times[-1]),
            "ref_date": str(times[0]),
        },
    )

    # Return as a new dataset
    return xr.Dataset({"velocity": vel_da})

def export_timeseries_to_geotiff(input_nc, frame_id="UNKNOWN", displacement_var="displacement", output_dir="export"):
    """
    Export time series NetCDF data to a multiband GeoTIFF with band metadata.

    Args:
        input_nc (str): Path to the input NetCDF file.
        frame_id (str): Frame ID to include in the filename.
        displacement_var (str): Variable name in the NetCDF to export (default 'displacement').
        output_dir (str): Directory where the output GeoTIFF will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_tif = os.path.join(output_dir, f"OPERA-DISP-S1_Frame_{frame_id}_displacement_timeseries.tif")

    ds = xr.open_dataset(input_nc)
    disp = ds[displacement_var]
    dates = ds["time"].dt.strftime('%Y-%m-%dT%H:%M:%S').values

    height, width = disp.shape[1], disp.shape[2]
    transform = from_origin(ds.x[0], ds.y[0], np.abs(ds.x[1] - ds.x[0]), np.abs(ds.y[1] - ds.y[0]))
    crs = ds.spatial_ref.attrs.get("crs_wkt") or ds.spatial_ref.attrs.get("spatial_ref")

    with rasterio.open(output_tif,"w",driver="GTiff",count=disp.shape[0],height=height,width=width,dtype=disp.dtype,crs=crs,transform=transform,nodata=np.nan,compress="DEFLATE",tiled=True,predictor=2,bigtiff="YES",zlevel=4,) as dst:
        for i in range(disp.shape[0]):
            dst.write(disp[i, :, :].values.astype(np.float32), i + 1)
            dst.update_tags(i + 1, DATE_TIME=str(dates[i]))

    with rasterio.open(output_tif, "r+") as ds_edit:
        for i, date in enumerate(dates):
            ds_edit.set_band_description(i + 1, f"Time series: {date}")

    return output_tif

def export_timeseries_pngs(input_nc, output_dir="export/pics", frame_id="UNKNOWN"):
    os.makedirs(output_dir, exist_ok=True)
    ds = xr.open_dataset(input_nc)
    disp = ds["displacement"]
    dates = ds["time"].dt.strftime('%Y%m%d').values
    ref_date = ds['reference_time'].dt.strftime('%Y%m%d').values[0]

    vmin_raw = np.nanpercentile(disp[-1], 1)
    vmax_raw = np.nanpercentile(disp[-1], 99)
    max_abs_val = max(abs(vmin_raw), abs(vmax_raw))
    vmin, vmax = -max_abs_val, max_abs_val

    file_paths = []
    for i, date in enumerate(dates):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(disp[i].values, cmap="jet", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(f"OPERA-DISP-S1 Frame {frame_id} -- Displacement from {dates[0]} to {date}")
        plt.colorbar(im, ax=ax, label="Displacement (m)")
        png_path = os.path.join(output_dir, f"OPERA-DISP-S1_Frame_{frame_id}_displacement_from_{ref_date}_to_{date}.png")
        plt.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        file_paths.append(png_path)

    return print('PNG Saved in /export/pics/')