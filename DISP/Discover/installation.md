# OPERA DISP-S1 Data Access and Time Series Analysis

This section sets up a full environment for accessing and processing Sentinel-1 displacement products from the OPERA DISP-S1 project.

It includes:
- Initializing and running a **Google Colab** environment (no local install needed)
- Installing the required **Conda environment** using **Miniforge**
- Downloading **Sentinel-1 displacement data** using `opera-utils`, filtered by frame ID and bounding box
- Reformatting the data into a **MintPy-compatible format** for time-series analysis

The setup works on Linux, macOS, and is fully compatible with Jupyter Notebooks and Google Colab.

*Prepared by: Bryan Raimbault*

---
## You can run this notebook directly in the cloud using Google Colab — no installation or Python handling required following the steps below:

### 0. Run a Google Colab 
You first need a Google account.  
Then, click the badge below to open the notebook and authorize the link between Google and Github to fetch the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OPERA-Cal-Val/OPERA_Applications/blob/main/DISP/Discover/Intro_to_DISP-S1.ipynb)

---

#### Open the notebook without linking GitHub (Alternative Option)

If you prefer **not to link Google and GitHub**, you can still open the notebook directly:

1. Go to [https://colab.research.google.com](https://colab.research.google.com)  
2. Click the **"GitHub"** tab *(do not log in)*  
3. Paste this raw notebook URL in the search box:
https://raw.githubusercontent.com/OPERA-Cal-Val/OPERA_Applications/main/DISP/Discover/Intro_to_DISP-S1.ipynb
4. Press Enter to open and run the notebook.
---
#### Drag & Drop Option (manual upload)

If you’ve already downloaded the notebook and input files:

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click the **"Upload"** tab
3. Drag and drop your `notebook.ipynb` file
4. Run each cell in sequence (Shift + Enter) and read the instructions

---
## You can also run the `Intro_to_DISP-S1.ipynb` notebook locally following the steps below:


### 1. Install Miniforge - Conda/Mamba
- **Linux/macOS**: Open the **Terminal** app.

```bash
mkdir -p /path/to/folder/tools; cd /path/to/folder/tools

# download, install and setup (mini)conda/mamba in the terminal
# for Linux:
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
# for macOS with Apple Silicon: 
curl https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o Miniforge3-MacOSX-arm64.sh
# for macOS with Intel: 
curl https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh -o Miniforge3-MacOSX-x86_64.sh
# Install Miniforge (adjust filename and installation path as needed)
bash Miniforge3-{Version}.sh -b -p /path/to/folder/tools/miniconda3
# Initialize conda for your shell: 
/path/to/folder/tools/miniconda3/bin/conda init bash
```

Close and restart the shell for changes to take effect.

```bash
# Add conda-forge channel
conda config --add channels conda-forge

# Install mamba (faster alternative to conda) and git
conda install git mamba --yes

# Create a new environment
conda create -n opera_disp-s1 --yes
conda activate opera_disp-s1

# Install required packages into the environment
mamba install -c conda-forge python==3.11.13 jupyter ipyleaflet gdal libgdal-netcdf --yes
```


### 2. Install OPERA DISP-S1 tools to `opera_disp-s1` environment

**Download source code:**

```bash
cd /path/to/folder/tools
git clone --depth 1 --no-checkout --branch main --no-checkout https://github.com/OPERA-Cal-Val/OPERA_Applications.git && cd OPERA_Applications
git sparse-checkout set DISP/Discover && git checkout && cd DISP/Discover
jupyter-notebook /path/to/folder/tools/OPERA_Applications/DISP/Discover/Intro_to_DISP-S1_Colab.ipynb
#This enables you to only clone the folder of interest opera_disp within the entire repository
```

### 3. Launching again the notebook in the future
```bash
# 1 Open a terminal
conda activate opera_disp-s1
jupyter-notebook /path/to/folder/tools/OPERA_Applications/DISP/Discover/Intro_to_DISP-S1_Colab.ipynb
```
---

### 4. Run the OPERA data downloading scripts:
If you want to download files directly without using Jupyter Notebooks, you can use the opera-utils command-line tool provided by the opera_utils Python package.

For example, here's how to download data for the Central Valley, California case study, using Sentinel-1 descending track 042 (Frame 11116). The full dataset for Frame 11116 contains approximately 300 interferograms (~102 GB total, ~340 MB per file). To reduce download time and storage usage, we recommend limiting the spatial with `--bbox` and temporal range using the `--start-datetime` and `--end-datetime arguments`.

```bash
# Arguments:
# --frame-id       OPERA frame number
# --output-dir     Folder to save downloaded data
# --bbox           Bounding box (lon_min lat_min lon_max lat_max)
# --start-datetime Start date (optional, format: YYYY-MM-DD)
# --end-datetime   End date (optional, format: YYYY-MM-DD)
# --num-workers    Number of parallel download workers

opera-utils disp-s1-download \
    --output-dir subset-ncs \
    --bbox -120.4397 36.0065 -120.1362 36.2195 \
    --frame-id 11116 \
    --start-datetime 2023-01-01 \
    --end-datetime 2023-01-31 \
    --num-workers 4
```

### 5. Reformat and Output Files to MintPy

This step takes the downloaded DISP-S1 displacement NetCDF (.nc) files and:

- Applies a referencing method to normalize displacement values (e.g., border median, high-coherence mask, or a specific point).
- Generates a new, single reformatted NetCDF output file (e.g., disp-output-**FRAME_ID**.nc).

```bash
# -----------------------------
# DISP-S1 Stack Reformat Script
# -----------------------------

# Set the reference method to one of the following:
#   "NONE"            → No referencing (raw displacement)
#   "POINT"           → Reference to a specific lat/lon location
#   "MEDIAN"          → Median over valid land pixels (excludes water)
#   "BORDER"          → Median over border pixels (configurable size)
#   "HIGH_COHERENCE"  → Median over high-coherence mask (based on threshold)

REFERENCE_METHOD="BORDER"  # Change this to your preferred method
FRAME_ID=11116
OUTPUT_NAME="disp-output-${FRAME_ID}.nc"

# Collect input files
INPUT_FILES=$(ls subset-ncs/*.nc | sort)
INPUT_FILES_STR=$(echo $INPUT_FILES)

# Run the DISP-S1 reformat command
opera-utils disp-s1-reformat \
    --drop-vars connected_component_labels shp_counts persistent_scatterer_mask timeseries_inversion_residuals short_wavelength_displacement \
    --reference-method $REFERENCE_METHOD \
    --output-name $OUTPUT_NAME \
    --reference-border-pixels 3 \
    --input-files $INPUT_FILES_STR

    # Optional reference parameters (uncomment as needed):
    # --reference-row <row> \
    # --reference-col <col> \
    # --reference-lat <lat> \
    # --reference-lon <lon> \
    # --reference-coherence-threshold 0.7
```

#### Convert output to MintPy format

This step converts the reprocessed displacement NetCDF file into a format compatible with [MintPy](https://github.com/insarlab/MintPy). It uses the first `.nc` file from the stack as a sample reference grid and generates standard MintPy outputs in a new `export/` folder.


The output includes:

- velocity.h5 - average linear velocity map (in m/year)
- timeseries.h5 - full displacement time series for each pixel (linked to the input NetCDF)
- avgSpatialCoh.h5 - average spatial coherence map (used for masking and visualization)
- disp-output-**FRAME_ID**.nc - the reprocessed OPERA displacement stack (moved into export/)

```bash
SAMPLE_FILE=$(echo $INPUT_FILES | awk '{print $1}')

python -m opera_utils.disp.mintpy $OUTPUT_NAME \
    --sample-disp-nc $SAMPLE_FILE \
    --outdir export

# Move the output to the export directory
mv $OUTPUT_NAME export/
```

### 6. How to view the data?
In a terminal, you can visualize the velocity.h5 / timeseries.h5 newly created with the MintPy tools.
```bash
## Need help with the arguments: tsview.py -h
tsview.py /path/to/work/folder/export/timeseries.h5

## Need help with the arguments: view.py -h
view.py /path/to/work/folder/export/velocity.h5
```
EOF