#!/usr/bin/env python3
"""Download OPERA DISP-S1 products and static layers for displacement analysis.

This script handles downloading of:
1. DISP-S1 netCDF files from CMR
2. CSLC static layer files from ASF
3. Associated geometry files

It supports version-specific downloads and handles burst ID mapping between frames.

Example:
    python run1_download_DISP_S1_Static.py --frameID 33039

Dependencies:
    asf_search, opera_utils,

Author: Jinwoo Kim, Simran S Sangha
February, 2025
"""
import argparse
import os, json
from datetime import datetime
from pathlib import Path

from datetime import datetime as dt

import asf_search as asf
import opera_utils
from opera_utils.disp._search import search
from opera_utils.geometry import stitch_geometry_layers
from opera_utils.download import L2Product
from mintpy.utils import utils as ut
from disp_xr import io,download
from rasterio.crs import CRS
import pandas as pd
import h5py
import xarray as xr
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from io import BytesIO
import zipfile
import netrc

import warnings
warnings.filterwarnings("ignore")

def createParser(iargs = None):
    '''Commandline input parser'''
    parser = argparse.ArgumentParser(description='Downloading OPERA DISP-S1 and static layer files from ASF')
    parser.add_argument("--frameID", 
                        required=True, type=str, help='frameID of DISP-S1 to download (e.g., 33039)')
    parser.add_argument("--bbox",
                        default=None, type=float, nargs=4, dest="bbox", help='Specify bounding box in lon/lat format: "min_lon max_lon min_lat max_lat"') 
    parser.add_argument("--dispDir",
                        default='OPERA_DISP_S1_Files', type=str, help='directory to download DISP-S1 (default: outputs)')
    parser.add_argument("--startDate", 
                        default=None, type=str, help='start date of DISP-S1 (default: None, YYYYMMDD)')
    parser.add_argument("--endDate", 
                        default=None, type=str, help='end date of DISP-S1 (default: None, YYYYMMDD)')
    parser.add_argument("--nWorkers",
                        default=5, type=int, help='number of simultaenous downloads (default: 5)')
    parser.add_argument("--staticDir",
                        default='static_lyrs', type=str, help='directory to store static layer files (default: static_lyrs)')
    parser.add_argument("--geomDir",
                        default='geometry', type=str, help='directory to store geometry files from static layers (default: geometry)')
    parser.add_argument("--burstDB-version", 
                        default='0.9.0', type=str, help='burst DB version (default: 0.9.0)')
    parser.add_argument("--staticOnly",
                        action='store_true', help='download only static layer files without nc files')
    return parser.parse_args(args=iargs)

def process_file(url, bbox, outdir, username, password):
    filename = url.split("/")[-1]
    base, ext = os.path.splitext(filename)
    outname = f"{outdir}/{base}.nc"

    if os.path.exists(outname):
        print(f"Skipped (exists): {filename}")
        return
    
    session = requests.Session()
    session.auth = (username, password)
    response = session.get(url)
    response.raise_for_status()
    file_bytes = BytesIO(response.content)

    if bbox is not None:
        with h5py.File(file_bytes, "r") as h5f:
            # Open and slice root data
            ds = xr.open_dataset(h5f, engine="h5netcdf")
            subset = ds.isel(y=slice(bbox[2], bbox[3]), x=slice(bbox[0], bbox[1]))
            subset.to_netcdf(outname)
            ds.close() 

            # Also subset and add /corrections data
            ds_corr = xr.open_dataset(h5f, engine="h5netcdf", group="corrections")
            corr_subset = ds_corr.isel(y=slice(bbox[2], bbox[3]), x=slice(bbox[0], bbox[1]))
            corr_subset.to_netcdf(outname, mode="a", group="corrections")
            ds_corr.close() 

            # Add metadata-only groups with xarray
            for group in ["identification", "metadata"]:
                try:
                    meta = xr.open_dataset(h5f, engine="h5netcdf", group=group)
                    meta.to_netcdf(outname, mode="a", group=group)
                    meta.close()
                except Exception as e:
                    print(f"Warning: Could not write {group}: {e}")

            # Append broken subgroups with h5py
            copy_group_h5py(h5f, outname, "metadata/reference_orbit")
            copy_group_h5py(h5f, outname, "metadata/secondary_orbit")
    else:
        with h5py.File(file_bytes, "r") as h5f:
            ds = xr.open_dataset(h5f, engine="h5netcdf")
            ds.to_netcdf(outname)
            ds.close()
            ds_corr = xr.open_dataset(h5f, engine="h5netcdf", group="corrections")
            ds_corr.to_netcdf(outname, mode="a", group="corrections")
            ds_corr.close()
            
            # Add metadata-only groups with xarray
            for group in ["identification", "metadata"]:
                try:
                    meta = xr.open_dataset(h5f, engine="h5netcdf", group=group)
                    meta.to_netcdf(outname, mode="a", group=group)
                    meta.close()
                except Exception as e:
                    print(f"Warning: Could not write {group}: {e}")

            # Append broken subgroups with h5py
            copy_group_h5py(h5f, outname, "metadata/reference_orbit")
            copy_group_h5py(h5f, outname, "metadata/secondary_orbit")

    print(f"Done: {filename}")

def extract_pixel_bbox_from_lalo(coord, lonlat_bbox):
    min_lon,  max_lon, min_lat, max_lat = lonlat_bbox
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
    t = pd.to_datetime(
        [
            metadata["reference_zero_doppler_start_time"],
            metadata["reference_zero_doppler_end_time"],
        ]
    )
    t_mid = t[0] + t.diff()[1] / 2
    total_seconds = (
        t_mid.hour * 3600 + t_mid.minute * 60 + t_mid.second + t_mid.microsecond / 1e6
    )
    metadata["CENTER_LINE_UTC"] = total_seconds

    # Clean up of metadata dicts
    for key in ["reference_datetime", "secondary_datetime"]:
        del metadata[key]

    return metadata

def main(inps):
    frameID = inps.frameID
    frameID = frameID.zfill(5)    # force frameID to have 5 digit number as string
    dispDir = inps.dispDir
    os.makedirs(dispDir, exist_ok='True')
    startDate = inps.startDate
    endDate = inps.endDate
    nWorkers = inps.nWorkers
    staticDir = inps.staticDir
    os.makedirs(staticDir, exist_ok='True')
    geomDir = inps.geomDir
    os.makedirs(geomDir, exist_ok='True')
    bbox_bounds = inps.bbox
    DB_ver = inps.burstDB_version

    if not inps.staticOnly:
        # Download DISP-S1 data from CMR
        print('Downloading DISP-S1 data from CMR... ')
        # Get Earthdata credentials from ~/.netrc
        auth_info = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        username, _, password = auth_info
        # Search for DISP-S1 files in the specified date range
        products_df = download.search(frame_id=frameID, start_datetime=startDate, end_datetime=endDate)
        # Store all URLs in a list
        nc_urls = [line.strip() for line in products_df['filename'].values if isinstance(line, str) and line.strip()]
        print(f'Number of DISP F{frameID} granules: {products_df.shape[0]}')
        
        # Size Estimation of the full stack without cropping
        session = requests.Session()
        session.auth = (username, password)
        response = session.get(nc_urls[0])
        response.raise_for_status()
        file_bytes = BytesIO(response.content)
        file_size = file_bytes.getbuffer().nbytes/ 1024**2
        print(f"DISP-S1 file size: {file_size:.2f} MB") 
        print(f"This will result in a stack of {(len(nc_urls) * file_size)/ 1024:.2f} Gb, without cropping")
        print(f"Please make sure you have enough space or consider cropping the stack")
        
        if inps.bbox is not None:
            # Extract metadata from the first file
            meta = get_metadata(file_bytes)
            coord = ut.coordinate(meta)
            y_slice, x_slice = extract_pixel_bbox_from_lalo(coord, bbox_bounds)

            # Get pixel coordinate bounds from slices
            row_start, row_end = y_slice.start, y_slice.stop
            col_start, col_end = x_slice.start, x_slice.stop
            bbox_bounds = [col_start, col_end, row_start, row_end]
            print(f"Integer slicing for bbox: X from {col_start} to {col_end}, Y from {row_start} to {row_end}")
        
        # Download
        print('OPERA DISP-S1 data download started on ...')
        with ThreadPoolExecutor(max_workers=nWorkers) as executor:
            future_to_url = {executor.submit(process_file, url, bbox_bounds, dispDir, username, password): 
                url for url in nc_urls}
            for future in as_completed(future_to_url):
                result = future.result()
        
    print('OPERA DISP-S1 data downloaded, moving to static layers... ')
    # Access json matching bursts to frame IDs without downloading
    repo_zip_url = f'https://github.com/opera-adt/burst_db/releases/download/v{DB_ver}/opera-s1-disp-{DB_ver}-frame-to-burst.json.zip'

    # Access the ZIP file
    response = requests.get(repo_zip_url)
    zip_data = BytesIO(response.content)

    # Extract the JSON file from the ZIP archive
    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        json_data = zip_ref.read(f'opera-s1-disp-{DB_ver}-frame-to-burst.json') 

    # Load the JSON data
    data = json.loads(json_data.decode('utf-8'))
    burst_ids = data['data'][frameID.lstrip('0')]['burst_ids']  # list of burst IDs within one frame ID

    # search CLSC Static Layer files
    product = L2Product.CSLC_STATIC

    results = asf.search(
        operaBurstID=list(burst_ids),
        processingLevel=product.value,
    )

    results.download(path=staticDir, processes=nWorkers)    # downloading static layers with simultaneous downloads

    list_static_files = [ Path(f'{staticDir}/{results[ii].properties["fileName"]}') for ii in range(len(results)) ] 

    print('number of static layer files to download: ', len(results))

    # generating los_east.tif and los_north.tif from downloaded static layers
    output_files = stitch_geometry_layers(list_static_files, output_dir=geomDir)

    print('Done')

if __name__ == '__main__':
    # load arguments from command line
    inps = createParser()

    print("==================================================================")
    print("        Downloading DISP-S1 and static layer files")
    print("==================================================================")
    
    # Run the main function
    main(inps)
