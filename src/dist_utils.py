# Library Imports
import os
import re
import math
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import rasterio as rio
import rioxarray
from osgeo import gdal
import numpy as np
import numpy.ma as ma
import folium
from shapely.geometry import shape, Point, Polygon
from shapely.ops import transform
import matplotlib as mpl
import matplotlib.pyplot as plt
from netrc import netrc
from pyproj import Transformer
from subprocess import Popen
from getpass import getpass
from http import cookiejar
from urllib import request
import tempfile

def check_netrc():
    '''
    Checks that user possesses necessary credentials for accessing Earthdata in .netrc file. If not present, user is prompted to 
    enter username and password, which are placed in a .netrc file in user's home directory. 
    '''
    # -----------------------------------AUTHENTICATION CONFIGURATION-------------------------------- #
    urs = 'urs.earthdata.nasa.gov'    # Earthdata URL to call for authentication
    prompts = ['Enter NASA Earthdata Login Username \n(or create an account at urs.earthdata.nasa.gov): ',
           'Enter NASA Earthdata Login Password: ']

    # Determine if netrc file exists, and if so, if it includes NASA Earthdata Login Credentials
    try:
        netrcDir = os.path.expanduser("~/.netrc")
        netrc(netrcDir).authenticators(urs)[0]
        print('netrc exists and includes NASA Earthdata login credentials.')
        
    # Below, create a netrc file and prompt user for NASA Earthdata Login Username and Password
    except FileNotFoundError:
        homeDir = os.path.expanduser("~")
        Popen('touch {0}.netrc | chmod og-rw {0}.netrc | echo machine {1} >> {0}.netrc'.format(homeDir + os.sep, urs), shell=True)
        Popen('echo login {} >> {}.netrc'.format(getpass(prompt=prompts[0]), homeDir + os.sep), shell=True)
        Popen('echo password {} >> {}.netrc'.format(getpass(prompt=prompts[1]), homeDir + os.sep), shell=True)

    # Determine OS and edit netrc file if it exists but is not set up for NASA Earthdata Login
    except TypeError:
        homeDir = os.path.expanduser("~")
        Popen('echo machine {1} >> {0}.netrc'.format(homeDir + os.sep, urs), shell=True)
        Popen('echo login {} >> {}.netrc'.format(getpass(prompt=prompts[0]), homeDir + os.sep), shell=True)
        Popen('echo password {} >> {}.netrc'.format(getpass(prompt=prompts[1]), homeDir + os.sep), shell=True)
    #return netrc(netrcDir).authenticators(urs)
    return 

def clipPercentile(x):
    '''
    Returns an array containing the original values clipped to the 2nd and 98th percentile of the input data.
    This provides more vivid visualizations of the data.
            Parameters:
                x (array): Numpy array containing the original data
            Returns:
                x_clipped (array): Numpy array containing the data clipped to values between the 2nd and 98th percentile of the input array.
    '''
    x_clipped = np.clip(x, np.nanpercentile(x, 2), np.nanpercentile(x,98))
    return(x_clipped)

def colorize(array, cmap='hot_r'):
    '''
    Converts pixels to RGB, adjusting colorscale relative to data range.
            Parameters:
                    array (numpy array): Array of pixels to assign RGB values
                    cmap (colormap): Colormap to assign to pixels
            Returns:
                Pixels with RGB values corresponding to the specified cmap.
    '''
    normed_data = (array - array.min()) / (array.max() - array.min()) 
    cm = plt.cm.get_cmap(cmap)
    return cm(normed_data) 

def compute_area(data_,bounds, pixel_area, ref_date):
    '''
    Returns area of wildfire extent for single day.
            Parameters:
                    data (numpy array): Dist Status values (1.0-4.0)
                    bounds (list): Boundary of the area of interest (pixel value)
                    pixel_area (float): Area of one pixel (m)
            Returns:
                    fire_area (str): Wildfire extent area in kilometers squared
    '''
    data = data_[bounds[0]:bounds[1], bounds[2]:bounds[3]]
    fire_pixel_count = len(data[np.where(data>0)])
    fire_area = fire_pixel_count * pixel_area * pow(10, -6)
    fire_area = str(math.trunc(fire_area)) + " kilometers squared"
    return fire_area

def compute_areas(stats, pixel_area, product='alert', date=None):

    '''

    '''
    dates = []
    classes = []

    if product == 'alert':
        description = ['No disturbance', 'Provisional < 50%', 'Confirmed < 50%', 'Provisional  ≥ 50%', 'Confirmed  ≥ 50%']
    elif product == 'ann':
        description = ['No Disturbance', 'Confirmed < 50%, Ongoing',
                       'Confirmed ≥ 50%, Ongoing', 'Confirmed < 50%, complete', 'Confirmed ≥ 50%, Complete']
    elif product not in ['alert', 'ann']:
        raise Exception("Invalid value for 'product'. It should be 'alert' or 'ann'.")
    
    areas_km = []
    areas_hectares = []
    
    for i in stats[0]:
        if date is not None:
            dates.append(date)
        else:
            dates.append(0)
        classes.append(i)
        areas_km.append(stats[0][i] * pixel_area * pow(10, -6))
        areas_hectares.append((stats[0][i] * pixel_area * pow(10, -6))*100)

    affected_areas = pd.DataFrame(
        {'Date': dates,
         'VEG-DIST-STATUS Class': classes,
         'Description': description,
         'Area (km2)': areas_km,
         'Area (hectares)': areas_hectares
        },
    )
    return(affected_areas)

def extract_date_from_string(input_string):
    '''
    This function is probably not universal for all DIST-ALERT data. 
    It should work for the data stored on GLAD in the SEP folder.
    '''
    # Define a regex pattern to match the date in the format 'yyyymmdd' or 'yyyddd'
    date_pattern = r'(\d{4}(?:\d{2}\d{2}|\d{3}))'

    # Search for the pattern in the input URL
    match = re.search(date_pattern, input_string)

    # If a match is found, extract and return the date
    if match:
        return match.group(1)
    else:
        return None
    
def getbasemaps():
    '''
    Add custom base maps to folium.
    '''
    basemaps = {
        'Google Maps': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Maps',
            overlay = False,
            control = True,
            show = False,
        ),
        'Google Satellite': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Satellite',
            overlay = True,
            control = True,
            #opacity = 0.8,
            show = False
        ),
        'Google Terrain': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Terrain',
            overlay = False,
            control = True,
            show = False,
        ),
        'Google Satellite Hybrid': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Satellite',
            overlay = True,
            control = True,
            #opacity = 0.8,
            show = False
        ),
        'Esri Satellite': folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri',
            name = 'Esri Satellite',
            overlay = True,
            control = True,
            #opacity = 0.8,
            show = False
        )
    }

    return basemaps

def get_pixel_value_at_coordinate(filepath, coords, output_epsg):
    '''
    Get pixel value for an input geotiff and coordinate location.
            Parameters:
                    filepath (url): Path to the location of the geotiff
                    coords (list): Coordinate lat,lon
                    output_epsg (int): EPSG code for the raster data
            Returns:
                Single pixel value (for a single banded raster)
    '''
    # Create a coordinate transformer
    EPSG_WGS84 = 4326

    transformer = Transformer.from_crs(EPSG_WGS84, output_epsg, always_xy=True)

    # Convert the input latitude and longitude to EPSG 3857
    x, y = transformer.transform(coords[1], coords[0])

    # Open the GeoTIFF file
    with rio.open(filepath) as dataset:

        # Get the row and column indices corresponding to the transformed coordinates
        row, col = dataset.index(x, y)
        
        # Read the pixel value at the computed row and column indices
        pixel_value = dataset.read(1, window=((row, row + 1), (col, col + 1)))

    return pixel_value[0][0]

def handle_draw(target, action, geo_json):
    '''
    Neccessary for returning user-defined AOI in interactive map
    '''
    return

def intersection_percent(item, aoi):
    '''
   Returns percentage that Item's geometry intersects the AOI.
            Parameters:
                    item (Iem): DIST tile
                    aoi (dict): Area of interest
            Returns:
                    intersection_percent (float): Percentage that Item's geometry intersects 
                                                    the AOI (An item that completely covers
                                                    the AOI has a value of 100)
    '''
    geom_item = shape(item.geometry)
    geom_aoi = shape(aoi)
    intersected_geom = geom_aoi.intersection(geom_item)
    intersection_percent = (intersected_geom.area * 100) / geom_aoi.area

    return intersection_percent

def make_veg_dist_status_visual(filepath, filename):
    '''
    Return a rendered visual of a VEG-DIST-STATUS tile.
            Parameters:
                filepath (url): Path to the location of the VEG-DIST-STATUS tile.
                filename (str): Output filename.
            Returns:
                No returns. Saves .tif file locally.
    '''
    print('making VEG-DIST-STATUS rendering...')

    # make output subdirectory, if not already present
    out_dir = 'tifs/'
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    with rio.open(filepath, mode='r') as src:
        transform = src.transform
        crs = src.crs
        meta = src.meta

    data = gdal.Open(filepath)
    array = data.GetRasterBand(1).ReadAsArray()
    data = None

    # Define colormapping
    color_mapping = {
        2: (255, 255, 178, 1),
        5: (244, 86, 41, 1),
        4: (254, 183, 81, 1),
        6: (189, 0, 38, 1),
        255: (0, 0, 0, 0),
        0: (255, 255, 255, 0)
    }

    with rio.open(out_dir+filename, 'w', **meta) as dst:
        dst.write(array, indexes=1)
        dst.write_colormap(1, color_mapping)
        nodata=0

    print(filename+' written successfully.')

    return

def make_hls_true_color(filepath, bandlist, filename):
    '''
    Return a rendered true color of an input HLS tile.
            Parameters:
                filepath (url): Path to the location of the HLS tile.
                bandlist (list): List of bands to produce the true color (bandlist may contain bands not required for true color).
                filename (str): Output filename.
            Returns:
                No returns. Saves .tif file locally.
    '''

    print('making hls true color rendering...')

    # make output subdirectory, if not already present
    out_dir = 'tifs/'
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    for i,b in enumerate(bandlist):
        band = filepath+b+'.tif'

        #get transform/crs
        if i == 0:
            with rio.open(band, mode='r') as src:
                transform = src.transform
                crs = src.crs
                
        #load bands  
        if b == "B04":
            data = gdal.Open(band)
            red = data.GetRasterBand(1).ReadAsArray()
        if b == "B03":
            data = gdal.Open(band)
            green = data.GetRasterBand(1).ReadAsArray() 
        if b == "B02":
            data = gdal.Open(band)
            blue = data.GetRasterBand(1).ReadAsArray()

    data = None

    redClipped = clipPercentile(red)
    greenClipped = clipPercentile(green)
    blueClipped = clipPercentile(blue)
                
    redClipped_scaled = scaleto255(redClipped)
    greenClipped_scaled = scaleto255(greenClipped)
    blueClipped_scaled = scaleto255(blueClipped)
    
    cube = np.stack((redClipped_scaled, blueClipped_scaled, greenClipped_scaled)).astype('uint8')

    RGB_dataset = rio.open(
        str(out_dir+filename),
        'w',
        driver='GTiff',
        height=cube.shape[1],
        width=cube.shape[2],
        count=3,
        dtype=cube.dtype,
        crs=crs,
        transform=transform
    )

    RGB_dataset.write(cube)
    RGB_dataset.close() 

    print(filename+' written successfully.')
    return                     

def make_hls_false_color(filepath, bandlist, filename):
    '''
    Return a rendered false color of an input HLS tile.
            Parameters:
                filepath (url): Path to the location of the HLS tile.
                bandlist (list): List of bands to produce the false color (bandlist may contain bands not required for false color).
                filename (str): Output filename.
            Returns:
                No returns. Saves .tif file locally.
    '''
    print('making false color rendering...')

    # make output subdirectory, if not already present
    out_dir = 'tifs/'
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    for i,b in enumerate(bandlist):
        band = filepath+b+'.tif'

        #get transform/crs
        if i == 0:
            with rio.open(band, mode='r') as src:
                transform = src.transform
                crs = src.crs
                
        #load bands  
        if b == "B05" or b =="B08":
            data = gdal.Open(band)
            nir = data.GetRasterBand(1).ReadAsArray()
        if b == "B03":
            data = gdal.Open(band)
            green = data.GetRasterBand(1).ReadAsArray() 
        if b == "B02":
            data = gdal.Open(band)
            blue = data.GetRasterBand(1).ReadAsArray()

    data = None

    nirClipped = clipPercentile(nir)
    greenClipped = clipPercentile(green)
    blueClipped = clipPercentile(blue)
                
    nirClipped_scaled = scaleto255(nirClipped)
    greenClipped_scaled = scaleto255(greenClipped)
    blueClipped_scaled = scaleto255(blueClipped)

    cube = np.stack((nirClipped_scaled, blueClipped_scaled, greenClipped_scaled)).astype('uint8')

    NBG_dataset = rio.open(
        str(out_dir+filename),
        'w',
        driver='GTiff',
        height=cube.shape[1],
        width=cube.shape[2],
        count=3,
        dtype=cube.dtype,
        crs=crs,
        transform=transform
    )

    NBG_dataset.write(cube)
    NBG_dataset.close() 

    print(filename+' written successfully.')
    return

def make_hls_ndvi(filepath, bandlist, filename):
    '''
    Return a rendered ndvi of an input HLS tile.
            Parameters:
                filepath (url): Path to the location of the HLS tile.
                bandlist (list): List of bands to produce the NDVI (bandlist may contain bands not required for NDVI).
                filename (str): Output filename.
            Returns:
                No returns. Saves .tif file locally.
    '''
    
    print('making ndvi rendering...')

    # make output subdirectory, if not already present
    out_dir = 'tifs/'
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    for i,b in enumerate(bandlist):
            
            band = filepath+b+'.tif'
            
            #get transform/crs
            if i == 0:
                with rio.open(band, mode='r') as src:
                    transform = src.transform
                    crs = src.crs  
            #load bands  
            if b == "B05" or b == "B08":
                data = gdal.Open(band)
                nir = data.GetRasterBand(1).ReadAsArray()
            elif b == "B04":
                data = gdal.Open(band)
                red = data.GetRasterBand(1).ReadAsArray()
            elif b == "B03":
                data = gdal.Open(band)
                green = data.GetRasterBand(1).ReadAsArray() 
            elif b == "B02":
                data = gdal.Open(band)
                blue = data.GetRasterBand(1).ReadAsArray()
    
    if (nir is not None) & (red is not None):

        #compute NDVI
        ndvi = (nir - red) / (nir + red)
        mask = (ndvi > 0) & (ndvi < 1)
        ndvi_cor = np.ma.masked_array(ndvi, ~mask)
        
        ndviClipped = clipPercentile(ndvi_cor)
        ndviClipped_scaled = scaleto255(ndviClipped)

        # Reshape the array
        cube = np.reshape(ndviClipped_scaled, (1, ndviClipped_scaled.shape[0], ndviClipped_scaled.shape[1])).astype('uint8')
        
        NDVI_dataset = rio.open(
            str(out_dir+filename),
            'w',
            driver='GTiff',
            height=cube.shape[1],
            width=cube.shape[2],
            count=1,
            dtype=cube.dtype,
            crs=crs,
            transform=transform
        )

        NDVI_dataset.write(cube)
        NDVI_dataset.close() 

        print(filename+' written successfully.')
        
    else:
        print('missing necessary bands to compute ndvi.')
    
    return

def mask_rasters(merged_VEG_ANOM_MAX, merged_VEG_DIST_DATE, merged_VEG_DIST_STATUS):
    '''
    Return VEG-ANOM-MAX, VEG-DIST-DATE, VEG-DIST-STATUS rasters with nan values masked.
            Parameters:
                    merged_VEG_ANOM_MAX (array): Merged VEG-ANOM-MAX arrays
                    merged_VEG_DIST_DATE (array): Merged VEG-DIST-DATE arrays
                    merged_VEG_DIST_STATUS (array): Merged VEG-DIST-STATUS arrays
            Returns:
                    masked_VEG_ANOM_MAX (array): merged_VEG_ANOM_MAX array with nan values masked
                    masked_VEG_DIST_DATE (array): merged_VEG_DIST_DATE array with nan values masked
                    masked_VEG_DIST_STATUS (array): merged_VEG_DIST_STATUS array with nan values masked
    '''
    raster_da_VEG_ANOM_MAX = merged_VEG_ANOM_MAX.where((merged_VEG_ANOM_MAX<=100), np.nan)
    arr_raster_da_VEG_ANOM_MAX = raster_da_VEG_ANOM_MAX.values
    masked_VEG_ANOM_MAX = ma.masked_invalid(arr_raster_da_VEG_ANOM_MAX)
    
    raster_da_VEG_DIST_DATE = merged_VEG_DIST_DATE.where((merged_VEG_DIST_DATE>0), np.nan)
    arr_raster_da_VEG_DIST_DATE = raster_da_VEG_DIST_DATE.values
    masked_VEG_DIST_DATE = ma.masked_invalid(arr_raster_da_VEG_DIST_DATE)
    
    raster_da_VEG_DIST_STATUS = merged_VEG_DIST_STATUS.where((merged_VEG_DIST_STATUS>0) & (merged_VEG_DIST_STATUS<255), np.nan)
    arr_raster_da_VEG_DIST_STATUS = raster_da_VEG_DIST_STATUS.values
    masked_VEG_DIST_STATUS = ma.masked_invalid(arr_raster_da_VEG_DIST_STATUS)
    
    return masked_VEG_ANOM_MAX, masked_VEG_DIST_DATE, masked_VEG_DIST_STATUS

def merge_rasters(input_files, output_file, bounds=None, write=True):
    """
    Function to take a list of raster tiles, mosaic them using rasterio, and output the file.
    :param input_files: list of input raster files 
    """
    from rasterio.merge import merge

    # Open the input rasters and retrieve metadata
    src_files = [rio.open(file) for file in input_files]
    meta = src_files[0].meta
    
    #mosaic the src_files
    if bounds is None:
        mosaic, out_trans = merge(src_files)
    else:
        mosaic, out_trans = merge(src_files, bounds)

    # Update the metadata
    out_meta = meta.copy()
    out_meta.update({"driver": "GTiff", 
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2], 
                    "transform": out_trans
                    }
                    )
    if write==True:
        with rio.open(output_file, 'w', **out_meta) as dst:
            dst.write(mosaic)

        #Close the input rasters
    for src in src_files:
        src.close()
    
    return mosaic

def scaleto255(x):
    '''
    Returns an array containing values between 0-255 that correspond to the original input array. 
    This array is essentially a rendered version of the input data scaled to 8-bit integer format.
            Parameters:
                x (array): Numpy array containing the original data
            Returns:
                x_scaled (array): Numpy array containing the data scaled to values between 0-255
    '''
    x_scaled = ((x - np.nanmin(x))) * (255/(np.nanmax(x)-np.nanmin(x)))
    return(x_scaled)

def stack_bands(bandpath:str, bandlist:list): 
    '''
    Returns geocube with three bands stacked into one multi-dimensional array.
            Parameters:
                    bandpath (str): Path to bands that should be stacked
                    bandlist (list): Three bands that should be stacked
            Returns:
                    bandStack (xarray Dataset): Geocube with stacked bands
                    crs (int): Coordinate Reference System corresponding to bands


            Updates: Changed load data library from xarray to rioxarray due to deprecation of xarray.open_rasterio().
            This required excluding the .scales method as well, which may cause problems, but I will wait and see.
    '''
    bandStack = []; bandS = []; bandStack_ = [];
    for i,band in enumerate(bandlist):
        if i==0:
            #bandStack_ = xr.open_rasterio(bandpath%band)
            bandStack_ = rioxarray.open_rasterio(bandpath%band)
            #crs = pyproj.CRS.to_epsg(pyproj.CRS.from_proj4(bandStack_.crs))
            crs = bandStack_.rio.crs.to_epsg()
            #bandStack_ = bandStack_ * bandStack_.scales[0]
            bandStack = bandStack_.squeeze(drop=True)
            bandStack = bandStack.to_dataset(name='z')
            bandStack.coords['band'] = i+1
            bandStack = bandStack.rename({'x':'longitude', 'y':'latitude', 'band':'band'})
            bandStack = bandStack.expand_dims(dim='band')  
        else:
            #bandS = xr.open_rasterio(bandpath%band)
            bandS = rioxarray.open_rasterio(bandpath%band)
            #bandS = bandS * bandS.scales[0]
            bandS = bandS.squeeze(drop=True)
            bandS = bandS.to_dataset(name='z')
            bandS.coords['band'] = i+1
            bandS = bandS.rename({'x':'longitude', 'y':'latitude', 'band':'band'})
            bandS = bandS.expand_dims(dim='band')
            bandStack = xr.concat([bandStack, bandS], dim='band')
    return bandStack, crs

def standard_date(day, ref_date):
    '''
    Returns the inputted day number as a standard date.
            Parameters:
                    day (str): Day number that should be converted
                    ref_date (datetime): Date of the beginning of the record
            Returns:
                    res (str): Standard date corresponding to inputted day
    '''
    day.rjust(3 + len(day), '0')
    res_date = ref_date + timedelta(days=int(day))
    res = res_date.strftime("%m-%d-%Y")
    return res

def time_and_area_cube(dist_status, dist_date, veg_anom_max, anom_threshold, pixel_area, bounds, starting_day, ending_day, ref_date, step=3):
    '''
    Returns geocube with time and area dimensions.
            Parameters:
                    dist_status (xarray DataArray): Disturbance Status band
                    anom_max (xarray DataArray): Maximum Anomaly band
                    dist_date (xarray DataArray): Disturbance Date band
                    anom_threshold (int): Filter out pixels less than the value
                    pixel_area (float): Area of one pixel (m)
                    bounds (list): Boundary of the area of interest (pixel value)
                    starting_day (int): First observation date
                    ending_day (int): Last observation date
                    ref_date (datetime): Date of the beginning of the record
                    step (int): Increment between each day in time series

            Returns:
                    wildfire_extent (xarray Dataset): Geocube with time and area dimensions
    '''
    lats = np.array(dist_status.latitude)
    lons = np.array(dist_status.longitude)
    expanded_array1 = []
    expanded_array2 = []
    respective_areas = {}

    for i in range(starting_day, ending_day, step):
        vg = dist_status.where((veg_anom_max > anom_threshold) & (dist_date > starting_day) & (dist_date <= i))
        extent_area = compute_area(vg.data,bounds,pixel_area, ref_date)
        date = standard_date(str(i), ref_date)
        coords =  {'lat': lats, 'lon': lons, 'time': date, 'area':extent_area}
        time_and_area = xr.DataArray(vg.data, coords=coords, dims=['lat', 'lon'])
        expanded_time_and_area = xr.concat([time_and_area], 'time')
        expanded_time_and_area = expanded_time_and_area.to_dataset(name='z')
        expanded_array2.append(expanded_time_and_area)
    area_extent = xr.concat(expanded_array2[:], dim='time')
    return area_extent

def transform_data_for_folium(url=[]):

    '''
    Returns data that is tranformed to be read correctly by Folium.
            Parameters:
                url (web url): Url to data location
            Returns:
                reproj (array): Array that is reprojected to EPSG 4326
                colormap (cmap): Colormap of choice
    '''
    src = rioxarray.open_rasterio(url)
    reproj = src.rio.reproject("EPSG:4326")             # Folium maps are in EPSG:4326
    colormap = mpl.colormaps["hot_r"]
    
    return reproj, colormap