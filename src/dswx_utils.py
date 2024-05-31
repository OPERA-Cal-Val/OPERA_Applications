from typing import Any, Dict

import boto3
import fnmatch
import folium
import fsspec
import numpy as np
import rasterio as rio
import rioxarray
from botocore import UNSIGNED
from botocore.client import Config
from matplotlib.colors import ListedColormap
from pystac.item import Item
from shapely.geometry import shape

# Query s3 for files
def query_s3_bucket(bucket_name, wildcard_pattern):
    '''Query s3 for all DSWx sample products
    '''
    # Initialize a session using an unsigned S3 client
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    paginator = s3.get_paginator('list_objects_v2')
    files = []

    for page in paginator.paginate(Bucket=bucket_name):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if fnmatch.fnmatch(key, wildcard_pattern):
                    files.append(f's3://{bucket_name}/{key}')
    
    return files

# Function to calculate percentage overlap between user-defined bbox and dswx tile
def intersection_percent(item: Item, aoi: Dict[str, Any]) -> float:
    '''The percentage that the Item's geometry intersects the AOI. An Item that
    completely covers the AOI has a value of 100.
    '''
    geom_item = shape(item.geometry)
    geom_aoi = shape(aoi)
    intersected_geom = geom_aoi.intersection(geom_item)
    intersection_percent = (intersected_geom.area * 100) / geom_aoi.area

    return intersection_percent

# Convert each pixels to RGBA for Folium
def colorize(array=[], cmap=[]):
    cmap[0] = (0, 0, 0, 0)              # Make zeroes transparent
    cm = ListedColormap([np.array(cmap[key]) / 255 for key in range(256)])
    
    return cm(array), cm

# Basemaps for Folium
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

# Transform the data to Folium projection
def transform_data_for_folium(og_url=[]):

    # handle properly if s3 links
    with fsspec.open(og_url, mode='rb', anon=True, default_fill_cache=False) as url:
        with rioxarray.open_rasterio(url) as src:
            reproj = src.rio.reproject("EPSG:4326")             # Folium maps are in EPSG:4326
        
        with rio.open(url) as ds:
            colormap = ds.colormap(1)

    return reproj, colormap
