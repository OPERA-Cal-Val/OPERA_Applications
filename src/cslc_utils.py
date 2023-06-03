import rasterio
from rasterio.crs import CRS
import h5py
import numpy as np
import fsspec
import matplotlib.pyplot as plt
import folium

def read_cslc(h5file):
    # Load the CSLC and necessary metadata
    grid_path = f'data'
    metadata_path = f'metadata'
    burstmetadata_path = f'metadata/processing_information/input_burst_metadata'
    id_path = f'identification'

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
    grid_path = f'data'
    metadata_path = f'metadata'
    burstmetadata_path = f'metadata/processing_information/input_burst_metadata'
    id_path = f'identification'

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
def colorize(array=[], cmap='RdBu', cmin=[], cmax=[]):
    normed_data = (array - cmin) / (cmax - cmin)    
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

def scale_amp(img):
    out = 20*np.log10(np.abs(img))
    
    return np.clip(out, np.nanmean(out)-np.nanstd(out),np.nanmean(out)+np.nanstd(out))

# Modified after https://github.com/opera-adt/dolphin/blob/4289be10e019da9c8da6c659370601ea96e63fe9/src/dolphin/interferogram.py#L509-L552
def moving_window_mean(image, size) -> np.ndarray:
    """Calculate the mean of a moving window of size `size`.

    Parameters
    ----------
    image : ndarray
        input image
    size : int or tuple of int
        Window size. If a single int, the window is square.
        If a tuple of (row_size, col_size), the window can be rectangular.

    Returns
    -------
    ndarray
        image the same size as `image`, where each pixel is the mean
        of the corresponding window.
    """
    if isinstance(size, int):
        size = (size, size)
    if len(size) != 2:
        raise ValueError("size must be a single int or a tuple of 2 ints")
    if size[0] % 2 == 0 or size[1] % 2 == 0:
        size = tuple(map(sum, zip(size, (1,1))))

    row_size, col_size = size
    row_pad = row_size // 2
    col_pad = col_size // 2

    # Pad the image with zeros
    image_padded = np.pad(
        image, ((row_pad + 1, row_pad), (col_pad + 1, col_pad)), mode="constant"
    )

    # Calculate the cumulative sum of the image
    integral_img = np.cumsum(np.cumsum(image_padded, axis=0), axis=1)
    if not np.iscomplexobj(integral_img):
        integral_img = integral_img.astype(float)

    # Calculate the mean of the moving window
    # Uses the algorithm from https://en.wikipedia.org/wiki/Summed-area_table
    window_mean = (
        integral_img[row_size:, col_size:]
        - integral_img[:-row_size, col_size:]
        - integral_img[row_size:, :-col_size]
        + integral_img[:-row_size, :-col_size]
    )
    window_mean /= row_size * col_size
    
    return window_mean