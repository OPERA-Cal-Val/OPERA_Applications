# OPERA_Applications

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/OPERA-Cal-Val/OPERA_Applications.git/main)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/OPERA-Cal-Val/OPERA_Applications/tree/main/?flush_cache=True)


This repository provides a collection of interactive notebooks to the OPERA Products: Dynamic Surface Water eXtent (DSWx), Land Disturbance (DIST), and Displacement products.

The OPERA DSWx product maps pixel-wise surface water detections using the Harmonized Landsat-8 Sentinel-2 A/B (HLS) data. More information about OPERA DSWx is available at https://www.jpl.nasa.gov/go/opera/products/dswx-product-suite. Also refer to the DSWx Product white paper [[here](https://d2pn8kiwq2w21t.cloudfront.net/documents/finalDSWx_URS306072_9n6sBVQ.pdf)] for high-level information.

The OPERA DIST product maps per pixel vegetation disturbance (specifically, vegetation cover loss) from the Harmonized Landsat-8 Sentinel-2 A/B (HLS) data. More information about OPERA DIST is available at https://www.jpl.nasa.gov/go/opera/products/dist-product-suite. Also refer to the DIST Product white paper [[here](https://d2pn8kiwq2w21t.cloudfront.net/documents/finalDIST_URS306040_a3pKEmP.pdf)] for high-level information.

The OPERA DISP product maps pixel-wise ground surface displacements along the radar line-of-sight from Sentinel-1 and NISAR dataset. This product currently under active development. More information about OPERA DISP will be released later. To learn more about the applications of DISP Product please refer to the following white papers: 1) [[hazards] [https://d2pn8kiwq2w21t.cloudfront.net/documents/finalDISP_hazards_URS306045.pdf]] and 2) [[monitoring] [https://d2pn8kiwq2w21t.cloudfront.net/documents/finalDISP_monitoring_URS306020_xtdvJS3.pdf]].

This repository includes directories for [DSWx](#dswx), [DIST](#dist), and [DISP](#disp). They contain several Jupyter notebooks that provide introductions and showcase applications of these products including flood mapping, water reservoir monitoring and monitoring wildfire evolution. To get started click the launch Binder logo above. Binder will open the Jupyter notebooks in an executable environment without requiring you to install any new software. 

## Contents
1. [Software Dependencies and Installation](#software-dependencies-and-installation)
3. [Jupyter Notebooks](jupyter-notebooks)
- [DSWx](#dswx)
    - [Discover](#discover)
    - [Flood](#flood)
    - [Reservoir](#reservoir)
- [DIST](#dist)
    - [Wildfire](#widlfire)
4. [Key Contributors](#key-contributors)

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
Please refer to each directory for notebooks for either DSWx or DIST.

### DSWx
There are several subdirectories dedicated to different types of applications for the OPERA DSWx products.

#### Discover
This directory contains Jupyter notebooks that showcase how to interface with DSWx products.

    .
    ├── ...
    ├── Discover                              
    │   ├── Stream_DSWx-HLS_HTTPSvsS3.ipynb                 # Access DSWx via HTTPs and S3
    │   ├── Stream_and_Viz_DSWx-HLS_viaCMR-STAC.ipynb       # Access DSWx via CMR-STAC
    │   └── Stream_and_Viz_DSWx-HLS_viaDirectHTTPS.ipynb    # Access DSWx via Direct HTTPS
    └── ...



#### Flood
This directory contains a Jupyter notebook that generates flood maps using provisional DSWx products over Pakistan.

    .
    ├── ...
    ├── Flood                             
    │   └── DSWx_FloodProduct.ipynb                # Create flood map using DSWx from the cloud
    └── ...

#### Reservoir
This directory contains Jupyter notebooks that demonstrate reservoir monitoring applications of provisional DSWx products over Lake Mead, NV. 

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
This includes a directory for wildfire applications.

#### Wildfire
This notebook contains Jupyter notebooks that demonstrate widlfire applicaitons of DIST products.

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
