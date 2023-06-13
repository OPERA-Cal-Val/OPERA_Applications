# OPERA Applications

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/OPERA-Cal-Val/OPERA_Applications.git/main)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/OPERA-Cal-Val/OPERA_Applications/tree/main/?flush_cache=True)


This repository provides a collection of interactive notebooks to the OPERA Products: Dynamic Surface Water eXtent (DSWx), Land Disturbance (DIST), and Displacement (DISP) products. They contain several Jupyter notebooks that provide introductions and showcase applications of these products including flood mapping, water reservoir monitoring and monitoring wildfire evolution. To get started click the launch Binder logo above. Binder will open the Jupyter notebooks in an executable environment without requiring you to install any new software. 

## Contents
1. [Software Dependencies and Installation](#software-dependencies-and-installation)
2. [Jupyter Notebooks](#jupyter-notebooks)
- [DSWx](#dswx)
    - [Discover](#discover)
    - [Flood](#flood)
    - [Reservoir](#reservoir)
    - [Mosaics](#mosaics)
- [DIST](#dist)
    - [Wildfire](#widlfire)
3. [Key Contributors](#key-contributors)

## Software Dependencies and Installation

This repository can be run by clicking on the Binder logo above or running on your local machine. For the required dependencies, we strongly recommend using [Anaconda](https://www.anaconda.com/products/distribution) package manager for easy installation of dependencies in the python environment. Below we outline how to access and manipulate this repository on your local machine using conda. <br>
First, download/clone the repository.
```
git clone https://github.com/OPERA-Cal-Val/OPERA_Applications
cd OPERA_Applications
```
Run the commands below to create a new conda environment `opera_app` and activate it:
```
conda env create -f environment.yml
conda activate opera_app
```
------
## Jupyter Notebooks
### DSWx
The OPERA DSWx product maps pixel-wise surface water detections using the Harmonized Landsat-8 Sentinel-2 A/B (HLS) data. More information about OPERA DSWx is available at https://www.jpl.nasa.gov/go/opera/products/dswx-product-suite. Also refer to the DSWx Product white paper [[here](https://d2pn8kiwq2w21t.cloudfront.net/documents/finalDSWx_URS306072_9n6sBVQ.pdf)] for high-level information.

Below describes the subdirectories within the DSWx folder.

#### Discover
This [discover directory](https://github.com/OPERA-Cal-Val/OPERA_Applications/tree/main/DSWx/Discover) contains Jupyter notebooks that showcase how to interface with DSWx products.

    .
    ├── ...
    ├── Discover                              
    │   ├── Stream_DSWx-HLS_HTTPSvsS3.ipynb                 # Access DSWx via HTTPs and S3
    │   ├── Stream_and_Viz_DSWx-HLS_viaCMR-STAC.ipynb       # Access DSWx via CMR-STAC
    │   └── Stream_and_Viz_DSWx-HLS_viaDirectHTTPS.ipynb    # Access DSWx via Direct HTTPS
    └── ...

#### Flood
The [flood directory](https://github.com/OPERA-Cal-Val/OPERA_Applications/tree/main/DSWx/Flood) contains a Jupyter notebook that generates flood maps using provisional DSWx products over Pakistan.

    .
    ├── ...
    ├── Flood                             
    │   └── DSWx_FloodProduct.ipynb                # Create flood map using DSWx from the cloud
    └── ...
    
#### Mosaics
This [mosaics directory](https://github.com/OPERA-Cal-Val/OPERA_Applications/tree/main/DSWx/Mosaics) demonstrates how PO.DAAC can be programmatically queried for DSWx data over a given region, for a specified time period. The returned DSWx granules are mosaicked to return a single raster image. As motivating examples, we demonstrate this over the state of California and the entireity of Australia.

    .
    ├── ...
    ├── Mosaics                              
    │   ├── notebooks
    │   │   └── Create-mosaics.ipynb           # Notebook to query PO.DAAC for DSWx data and mosaic returned granules
    │   ├── data
    │   │   ├── shapefiles                     # Shapefiles used to query PO.DAAC
    │   │   ├── australia                      # Folder containing example mosaicked raster over Australia
    │   │   └── california                     # Folder containing example mosaicked raster over CA
    │   ├── README.md
    │   └── environment.yml                    # YAML file containing dependencies needed to run code in this folder
    └── ...

#### Reservoir
This [reservoir directory](https://github.com/OPERA-Cal-Val/OPERA_Applications/tree/main/DSWx/Reservoir) contains Jupyter notebooks that demonstrate reservoir monitoring applications of provisional DSWx products over Lake Mead, NV. 

    .
    ├── ...
    ├── Reservoir                              
    │   ├── Intro_to_DSWx.ipynb                # Highlights four main layers of DSWx products
    │   ├── Reservoir_Monitoring.ipynb         # Reservoir monitoring of Lake Mead, NV between 2014-2022
    │   ├── Time_Slider_Visualization.ipynb    # Visualization of DSWx of Lake Mead, NV for the year 2022
    │   └── aux_files
    │       ├── T11SQA_manifest.txt            # S3 links to provisional products
    │       ├── prep_shapefile.ipynb           # Create buffered shapefile
    │       ├── lakebnds/                      # 2003 Lake Mead lake bounds shapefile
    │       └── bufferlakebnds/                # Buffered Lake Mead lake bounds shapefile
    └── ...

### DIST
The OPERA DIST product maps per pixel vegetation disturbance (specifically, vegetation cover loss) from the Harmonized Landsat-8 Sentinel-2 A/B (HLS) data. More information about OPERA DIST is available at https://www.jpl.nasa.gov/go/opera/products/dist-product-suite. Also refer to the DIST Product white paper [[here](https://d2pn8kiwq2w21t.cloudfront.net/documents/finalDIST_URS306040_a3pKEmP.pdf)] for high-level information.

Below describes the subdirectories within the DIST folder.

#### Discover
This [discover directory](https://github.com/OPERA-Cal-Val/OPERA_Applications/tree/main/DIST/Discover) contains Jupyter notebooks that showcase how to interface with DIST products.

    .
    ├── ...
    ├── Discover                              
    │   ├── Stream_and_Viz_DIST-ALERT-folium.ipynb    # Access DIST-ALERT via CMR-STAC
    │   └── Stream_and_Viz_DIST_Functions.py          # DIST functions
    └── ...

#### Wildfire
This [wildfire directory](https://github.com/OPERA-Cal-Val/OPERA_Applications/tree/main/DIST/Wildfire) contains Jupyter notebooks that demonstrate widlfire applicaitons of DIST products.

    .
    ├── ...
    ├── Wildfire                              
    │   ├── Intro_to_DIST.ipynb                # Highlights three main layers of DIST products
    │   ├── McKinney.ipynb                     # Visualization of 2022 McKinney wildfire with DIST
    │   └── aux_files
    │       └── McKinney_NIFC                  # Perimeter of 2022 McKinney wildfire
    └── ...
    
------
## Key Contributors
* Mary Grace Bato
* Kelly Devlin
* Rubie Dhillon
* Karthik Venkataramani
