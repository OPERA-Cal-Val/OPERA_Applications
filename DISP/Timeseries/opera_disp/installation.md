## Install OPERA DISP-S1 data access and preparation environment

Instructions derived and modified from https://github.com/nisar-solid/ATBD/blob/main/docs/installation.md and 
https://github.com/OPERA-Cal-Val/calval-DISP/blob/main/docs/installation.md/

Prepared by: Bryan Raimbault,
             Alexander Handwerger,
             Grace Bato
             Simran Sangha,
             Jin Woo Kim


### 1. Install Miniforge - Conda/Mamba

```bash
mkdir -p /path/to/folder/tools; cd /path/to/folder/tools

# download, install and setup (mini)conda/mamba
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
conda config --add channels conda-forge
conda install git mamba --yes
```

### 2. Install OPERA DISP-S1 tools to `opera_disp` environment

#### Download source code 

```bash
cd /path/to/folder/tools
git clone --depth 1 --no-checkout --branch main --no-checkout https://github.com/OPERA-Cal-Val/OPERA_Applications.git && cd OPERA_Applications
git sparse-checkout set DISP/Timeseries && git checkout && cd DISP/Timeseries
#This enables you to only clone the folder of interest opera_disp within the entire repository
```

#### Create `opera_disp` environment and install pre-requisites

```bash
cd /path/to/folder/tools/OPERA_Applications/DISP/Timeseries/opera_disp
# create new environment
# install dependencies with mamba by using `environment.yml`
mamba env create -f environment.yml
# load the environnement disp
conda activate opera_disp
```

#### Source your installation

Create a file (_e.g._: config.rc) for easy activation and loading of the paths to your files:

```bash
# creation of a empty file
touch /path/to/folder/tools/OPERA_Applications/DISP/Timeseries/opera_disp/config.rc
```
Add the following paths within the config.rc file:
```bash
##----------------------- OPERA DISP -----------------------##
# add repo tools to your path
export TOOL_DIR=/path/to/folder/tools
export PATH=${PATH}:${TOOL_DIR}/OPERA_Applications/DISP/Timeseries/opera_disp
export DISP_HOME=${TOOL_DIR}/OPERA_Applications/DISP/Timeseries/opera_disp
export PYTHONPATH=${PYTHONPATH}:${DISP_HOME}
```
Create an alias `load_disp` in `~/.bash_profile` file for easy activation, that call the config.rc file _e.g._:
```bash
alias load_disp='conda activate opera_disp; source /path/to/folder/tools/OPERA_Applications/DISP/Timeseries/opera_disp/config.rc'
#Close and restart the terminal for changes to take effect
```

### 3. Update the `opera_disp` environment MintPy packages

#### Install MintPy from source

```bash
# Load your environnement and paths
load_disp
cd /path/to/folder/tools/OPERA_Applications/DISP/Timeseries/opera_disp
git clone https://github.com/insarlab/MintPy.git
python -m pip install -e MintPy
```

#### Install disp-xr tool from source
```bash
# Load your environnement and paths
load_disp
cd /path/to/folder/tools/OPERA_Applications/DISP/Timeseries/opera_disp
git clone https://github.com/opera-adt/disp-xr.git
pip install -e disp-xr
```
### 4. Prepare credentials or register for NASA Earthdata access

1. Register for an account with NASA Earthdata at https://urs.earthdata.nasa.gov/users/new
2. After creating the username and confirming your email, store your username/password in a `~/.netrc` file with the hostname `urs.earthdata.nasa.gov`:
```
machine urs.earthdata.nasa.gov
  login MYUSERNAME
  password MYPASSWORD
```

## Troubleshooting Advice

If you encounter errors during usage, the most effective solution is to **"quit, re-open the terminal, and relaunch the Conda environment"**. This approach has successfully resolved the issue in all cases we've tested.

## Test the installation

Run the following to test the installation:

```bash
# Load OPERA displacement module
load_disp 

# Display help for the download script (try using 'python' if issues occur)
run1_download_DISP_S1_Static.py --h 

# Display help for MintPy
smallbaselineApp.py -h
```

### 5. Available OPERA DISP-S1 datasets are from 20160101 to 20241231:

You can look at your area of interest (AOI), using the OPERA_DISP_S1_Frames_viewer.ipynb or open the Frame_viewer.html in your local browser to find a OPERA DISP-S1 Frame of interest and extract the corresponding Frame ID.
You can also select one frame and toggling Ascending or Descending geometries:
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/OPERA-Cal-Val/OPERA_Applications/blob/notebook_disp/DISP/Timeseries/opera_disp/OPERA_DISP_S1_Frames_viewer.ipynb?flush_cache=true)


### 6. Run the OPERA data downloading script:
For example, here is a sample run for the Central Valley, California case study for descending Sentinel-1 track 042. 

For the Frame 11116, the size of the entire dataset of 300 files is ~102Gb, ~340Mb for a file. By default, the script processes all available dates, which may require substantial storage and processing time. To reduce the dataset size, you can select a bounding box. This will crop all the OPERA DISP-S1 files you are downloading according to your box and write the corresponding subsetted files for the bounding box with a reduced size.
```bash
# Args:
# --frameID    OPERA frame number
# --staticDir  Folder for static layers/metadata
# --geomDir    Folder for geometry files
# --dispDir    Folder for data
# --startDate  Start date (optional)
# --endDate    End date (optional)

run1_download_DISP_S1_Static.py \
      --frameID 11116 \
      --staticDir /path/to/work/folder/static_lyrs \
      --geomDir /path/to/work/folder/geometry \
      --dispDir /path/to/work/folder/data #\
      --bbox lon_min lon_max lat_min lat_max
     #--startDate 20170101
     #--endDate 20190101
```
### 7. Run the MintPy output script

For example, here is a sample run for the Central Valley, California case study for descending Sentinel-1 track 042.
You need to choose a spatial reference point for your dataset, as InSAR measurements are relative in both space and time. This point will be considered a fixed position in your dataset, and all other points will be measured relative to it. Be careful when choosing this reference point. For example, if you select a location that is actively moving (e.g., subsiding or uplifting), this motion will be reflected across the entire dataset and may potentially distort your interpretation.

```bash
## Example Command to Run `run2_prep_mintpy_opera.py`
# Args:
# -m   Folder for static layers/metadata
# -u   Folder with data (*.nc for all files)
# -g   Folder for geometry files
# -o   Folder for timeseries output
# --water-mask-file  Water mask file (auto-generated)
# --dem-file         DEM file (auto-generated)
# --ref-lalo         Spatial reference for timeseries
# --apply-mask       Apply mask (optional)

run2_prep_mintpy_opera.py \
        -m "/path/to/work/folder/static_lyrs" \
        -u "/path/to/work/folder/data/*.nc" \
        -g "/path/to/work/folder/geometry" \
        -o /path/to/work/folder/mintpy_output \
        --water-mask-file esa_world_cover_2021 \
        --dem-file glo_30 \
        --ref-lalo '36.612 -121.064' \
        --apply-mask
```

Note: 
`--apply-mask` applies the `recommended_mask` layer that is embedded within each of the DISP-S1 nominal product (i.e. *.nc) **on an epoch based**. The `recommended_mask` is the suggested mask to remove low quality pixels, where 0 indicates a bad pixel, 1 is a good pixel.

### 8. How to view the data?
In a terminal, you can visualize the timeseries.h5 newly created with the MintPy tools.
```bash
## Need help with the arguments: tsview.py -h
tsview.py /path/to/work/folder/mintpy_output/timeseries.h5 \
        -m /path/to/work/folder/mintpy_output/recommended_mask90threshold.h5 \
```

Note: 
`recommended_mask90threshold.h5` is based on the time-series of `recommended_mask` layers (i.e. `recommended_mask.h5`). We picked the top 90% representing the "most reliable pixels in time" after normalizing the `recommended_mask` against the total number of epoch/dataset. 
