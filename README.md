# OPERA_Applications

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/OPERA-Cal-Val/OPERA_Applications.git/main)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/OPERA-Cal-Val/OPERA_Applications/tree/main/)


This repo provides an interactive introduction to the OPERA Dynamic Surface Water eXtent (DSWx) and Land Disturbance (DIST) products. 

The OPERA DSWx (DSWx) product maps pixel-wise surface water detections using the Harmonized Landsat-8 Sentinel-2 A/B (HLS) images. More information about OPERA DSWx is available at https://www.jpl.nasa.gov/go/opera/products/dswx-product-suite. Also refer to the DSWx Product white paper[whitepaperlink] for high-level information.

The OPERA DIST product maps per pixel vegetation disturbance (specifically, vegetation cover loss) from the Harmonized Landsat-8 Sentinel-2 A/B (HLS) images. More information about OPERA DIST is available at https://www.jpl.nasa.gov/go/opera/products/dist-product-suite. Also refer to the DIST Product white paper [![here](https://d2pn8kiwq2w21t.cloudfront.net/documents/finalDIST_URS306040.pdf)] for high-level information.

This repo includes directories for both [DSWx](#dswx) and [DIST](#dist). They contain several Jupyter notebooks that provide introductions and showcase applications of these products including reservoir monitoring and wildfire evolution. To get started click the launch Binder logo above. Binder will open the Jupyter notebooks in an executable environment without requiring you to install any new software. 

## Contents
1. [Software Dependencies and Installation](#software-dependencies-and-installation)
3. [Jupyter Notebooks](jupyter-notebooks)
- [DSWx](#dswx)
    - [Intro to DSWx](#intro-to-dswx)
    - [Time slider visualization](#time-slider-visualization)
    - [Reservoir monitoring](#reservoir-monitoring)
- [DIST](#dist)
4. [Contributors](#contributors)

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
This includes a directory for reservoir monitoring applications.

#### Intro to DSWx
This notebook highlights four main layers of DSWx products and gives sample visualization of Lake Mead, NV.
+ [Intro to DSWx](link)

#### Time slider visualization
This notebook gives example of a time slider visualization of DSWx of Lake Mead, NV for the year 2022.
+ [Time slider visualization](link)

#### Reservoir monitoring
This notebook gives examples of DSWx applications for reservoir monitoring of Lake Mead, NV between 2014-2022.
+ [Reservoir monitoring](link)

### DIST

------
## Contributors
* Mary Grace Bato
* Matthew Bonnema
* Kelly Devlin
* Rubie Dhillon
* Simran Sangha
