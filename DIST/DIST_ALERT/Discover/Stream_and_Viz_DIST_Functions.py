#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Notebook dependencies
from pystac.item import Item
from typing import Dict, Any

from datetime import datetime, timedelta
import math

from shapely.geometry import shape
import numpy as np
import numpy.ma as ma
import rasterio as rio
import rioxarray
import xarray as xr
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pyproj
from pyproj import Proj

import folium


import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Global variables
total_pixel_count = 3660 * 3660
pixel_area = 30 * 30


# In[4]:


def intersection_percent(item: Item, aoi: Dict[str, Any]) -> float:
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


# In[5]:


def stack_bands(bandpath:str, bandlist:list): 
    '''
    Returns geocube with three bands stacked into one multi-dimensional array.
            Parameters:
                    bandpath (str): Path to bands that should be stacked
                    bandlist (list): Three bands that should be stacked
            Returns:
                    bandStack (xarray Dataset): Geocube with stacked bands
                    crs (int): Coordinate Reference System corresponding to bands
    '''
    bandStack = []; bandS = []; bandStack_ = [];
    for i,band in enumerate(bandlist):
        if i==0:
            bandStack_ = xr.open_rasterio(bandpath%band)
            crs = pyproj.CRS.to_epsg(pyproj.CRS.from_proj4(bandStack_.crs))
            bandStack_ = bandStack_ * bandStack_.scales[0]
            bandStack = bandStack_.squeeze(drop=True)
            bandStack = bandStack.to_dataset(name='z')
            bandStack.coords['band'] = i+1
            bandStack = bandStack.rename({'x':'longitude', 'y':'latitude', 'band':'band'})
            bandStack = bandStack.expand_dims(dim='band')  
        else:
            bandS = xr.open_rasterio(bandpath%band)
            bandS = bandS * bandS.scales[0]
            bandS = bandS.squeeze(drop=True)
            bandS = bandS.to_dataset(name='z')
            bandS.coords['band'] = i+1
            bandS = bandS.rename({'x':'longitude', 'y':'latitude', 'band':'band'})
            bandS = bandS.expand_dims(dim='band')
            bandStack = xr.concat([bandStack, bandS], dim='band')
            
    return bandStack, crs


# In[6]:


def time_and_area_cube(dist_max_anom, dist_date, dist_status, starting_day, ending_day, step=3):
    '''
    Returns geocube with time and area dimensions.
            Parameters:
                    dist_status (xarray DataArray): Disturbance Status band
                    anom_max (xarray DataArray): Maximum Anomaly band
                    dist_date (xarray DataArray): Disturbance Date band
                    starting_day (int): First day of interest (as number of days after 12-31-2020)
                    ending_day (int): Last day of interest (as number of days after 12-31-2020)
                    step (int): Increment between each day in time series
            Returns:
                    wildfire_extent (xarray Dataset): Geocube with time and area dimensions
    '''
    lats = np.array(dist_status.latitude)
    lngs = np.array(dist_status.longitude)
    expanded_array1 = []
    expanded_array2 = []
    respective_areas = {}
    
    for i in range(starting_day, ending_day, step):
        vg = dist_status
        vg = dist_status.where((dist_max_anom > 0) & (dist_date > starting_day) & (dist_date <= i))
        extent_area = compute_area(vg.data)
        date = standard_date(str(i))
        coords =  {'lat': lats, 'lng': lngs, 'time': date, 'area':extent_area}
        time_and_area = xr.DataArray(vg.data, coords=coords, dims=['lat', 'lng'])
        expanded_time_and_area = xr.concat([time_and_area], 'time')
        expanded_time_and_area = expanded_time_and_area.to_dataset(name='z')
        expanded_array2.append(expanded_time_and_area)
    area_extent = xr.concat(expanded_array2[:], dim='time')
    
    return area_extent


# In[7]:


def compute_area(data):
    '''
    Returns area of wildfire extent for single day.
            Parameters:
                    data (numpy array): Dist Status values (1.0-4.0)
            Returns:
                    fire_area (str): Wildfire extent area in kilometers squared
    '''
    nan_arr = np.isnan(data)
    nan_pixel_count = np.count_nonzero(nan_arr)
    fire_pixel_count = total_pixel_count - nan_pixel_count
    fire_area = fire_pixel_count * pixel_area * pow(10, -6)
    fire_area = str(math.trunc(fire_area)) + " kilometers squared"
    return fire_area


# In[8]:


def standard_date(day):
    '''
    Returns the inputted day number as a standard date
            Parameters:
                    day (str): Day number that should be converted
            Returns:
                    res (str): Standard date corresponding to inputted day
    '''
    init_strt_date = datetime(2021, 1, 1)
    day.rjust(3 + len(day), '0')
    res_date = init_strt_date + timedelta(days=int(day) - 1)
    res = res_date.strftime("%m-%d-%Y")
    return res


# In[9]:


# Convert each pixel to RGB, adjusting colorscale relative to data range
def colorize(array, cmap='hot_r'):
    normed_data = (array - array.min()) / (array.max() - array.min())    
    cm = plt.cm.get_cmap(cmap)
    return cm(normed_data) 


# In[10]:


# Mask individual bands
def mask_rasters(merged_VEG_ANOM_MAX, merged_VEG_DIST_DATE, merged_VEG_DIST_STATUS):
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


# In[11]:


def getbasemaps():
    # Add custom base maps to folium
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


# In[12]:


# Transform the data to Folium projection
def transform_data_for_folium(url=[]):
    src = rioxarray.open_rasterio(url)
    reproj = src.rio.reproject("EPSG:4326")             # Folium maps are in EPSG:4326
    colormap = mpl.colormaps["hot_r"]
    
    return reproj, colormap

