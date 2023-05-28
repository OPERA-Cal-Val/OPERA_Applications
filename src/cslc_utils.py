import rasterio
from rasterio.crs import CRS
import h5py
import numpy as np
import fsspec
import matplotlib.pyplot as plt

def read_cslc(h5file):
    # Load the CSLC and necessary metadata
    DATA_ROOT = 'science/SENTINEL1'
    grid_path = f'{DATA_ROOT}/CSLC/grids'
    metadata_path = f'{DATA_ROOT}/CSLC/metadata'
    burstmetadata_path = f'{DATA_ROOT}/CSLC/metadata/processing_information/s1_burst_metadata'
    id_path = f'{DATA_ROOT}/identification'

    if h5file[:2] == 's3':
        print(f'Streaming: {h5file}')  
        s3f = fsspec.open(h5file, mode='rb', anon=True, default_fill_cache=False)
        with h5py.File(s3f.open(),'r') as h5:
            cslc = h5[f'{grid_path}/VV'][:]
    else:
        with h5py.File(h5file,'r') as h5:
            print(f'Opening: {h5file}')  
            cslc = h5[f'{grid_path}/VV'][:]
        
    return cslc

def cslc_info(h5file):
    # Load the CSLC and necessary metadata
    DATA_ROOT = 'science/SENTINEL1'
    grid_path = f'{DATA_ROOT}/CSLC/grids'
    metadata_path = f'{DATA_ROOT}/CSLC/metadata'
    burstmetadata_path = f'{DATA_ROOT}/CSLC/metadata/processing_information/s1_burst_metadata'
    id_path = f'{DATA_ROOT}/identification'

    if h5file[:2] == 's3':
        s3f = fsspec.open(h5file, mode='rb', anon=True, default_fill_cache=False)
        with h5py.File(s3f.open(),'r') as h5:
            xcoor = h5[f'{grid_path}/x_coordinates'][:]
            ycoor = h5[f'{grid_path}/y_coordinates'][:]
            dx = h5[f'{grid_path}/x_spacing'][()].astype(int)
            dy = h5[f'{grid_path}/y_spacing'][()].astype(int)
            epsg = h5[f'{grid_path}/projection'][()].astype(int)
            bounding_polygon =h5[f'{id_path}/bounding_polygon'][()].astype(str) 
            orbit_direction = h5[f'{id_path}/orbit_pass_direction'][()].astype(str)         
    else:
        with h5py.File(h5file,'r') as h5:
            xcoor = h5[f'{grid_path}/x_coordinates'][:]
            ycoor = h5[f'{grid_path}/y_coordinates'][:]
            dx = h5[f'{grid_path}/x_spacing'][()].astype(int)
            dy = h5[f'{grid_path}/y_spacing'][()].astype(int)
            epsg = h5[f'{grid_path}/projection'][()].astype(int)
            bounding_polygon =h5[f'{id_path}/bounding_polygon'][()].astype(str) 
            orbit_direction = h5[f'{id_path}/orbit_pass_direction'][()].astype(str)
        
    return xcoor, ycoor, dx, dy, epsg, bounding_polygon, orbit_direction

def rasterWrite(outtif,arr,transform,epsg,dtype='float32'):
    #writing geotiff using rasterio
    
    new_dataset = rasterio.open(outtif, 'w', driver='GTiff',
                            height = arr.shape[0], width = arr.shape[1],
                            count=1, dtype=dtype,
                            crs=CRS.from_epsg(epsg),
                            transform=transform,nodata=np.nan)
    new_dataset.write(arr, 1)
    new_dataset.close() 

def custom_merge(old_data, new_data, old_nodata, new_nodata, **kwargs):    
    mask = np.logical_and(~old_nodata, ~new_nodata)
    old_data[mask] = new_data[mask]
    mask = np.logical_and(old_nodata, ~new_nodata)
    old_data[mask] = new_data[mask]

# Convert each pixel to RGB, adjusting colorscale relative to data range
def colorize(array, cmap='RdBu'):
    normed_data = (array - array.min()) / (array.max() - array.min())    
    cm = plt.cm.get_cmap(cmap)
    return cm(normed_data) 

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