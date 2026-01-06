# OPERA DISP-S1 Time Series Data for Landslides

This repository contains a Jupyter Notebook designed to demonstrate how to download and visualize surface displacement time series data from the **OPERA DISP-S1** product.

## Overview

The **OPERA Level-3 Surface Displacement from Sentinel-1 (DISP-S1)** product provides geocoded displacement measurements over North America. This notebook focuses on tracking cm-scale surface motion for slow-moving landslides.

### Workflow
1.  **Environment Setup**: Install necessary Python packages and helper tools.
2.  **Area Selection (AOI)**: Choose your target location and frame.
3.  **Data Download**: Pull cropped displacement data directly from NASA Earthdata.
4.  **Interactive Analysis**: Select a reference point on a map to adjust measurements.
5.  **Quality Review**: Inspect coherence and other metrics to verify data reliability.
6.  **Export**: Save results as GeoTIFFs for GIS or GIFs for animations.

---

## Getting Started

### 1. Conda Environment Setup
To run this notebook, you need a specific environment with InSAR analysis tools. Run the following commands in your terminal:

```bash
# 1. Create the environment
conda create -n opera_disp-s1

# 2. Activate it
conda activate opera_disp-s1

# 3. Install core dependencies
conda install -c conda-forge python==3.12 jupyter ipyleaflet
```

### 2. Automatic Dependency Check
The first code cell in the notebook will automatically check for and install remaining libraries, including:
- [MintPy](https://github.com/insarlab/MintPy)
- [disp-xr](https://github.com/opera-adt/disp-xr)
- Other required Python packages via `pip`.

### 3. Usage
Open the notebook in Jupyter:
```bash
jupyter-notebook OPERA-DISP-S1_Landslides.ipynb
```
Follow the step-by-step instructions provided directly within the notebook cells.

---

## Contributors
A. Handwerger, B. Raimbault, M. G. Bato, S. Sangha, M. Govorcin, S. Staniewicz.

