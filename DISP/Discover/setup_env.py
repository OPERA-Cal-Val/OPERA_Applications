# displacement_tools.py

import os
import sys
import subprocess

def run(command):
    """Run a shell command silently."""
    subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
def patch_version_constraints():
    """Update version constraints in pyproject.toml and environment.yml for Python 3.11.13."""

    # Patch pyproject.toml
    pyproject_path = "disp-xr/pyproject.toml"
    if os.path.exists(pyproject_path):
        with open(pyproject_path, "r") as f:
            lines = f.readlines()
        with open(pyproject_path, "w") as f:
            for line in lines:
                if line.strip().startswith("requires-python"):
                    f.write('requires-python = ">=3.11.13"\n')
                else:
                    f.write(line)

    # Patch environment.yml
    env_path = "environment.yml"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
        with open(env_path, "w") as f:
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("- python") or stripped.startswith("python"):
                    f.write("  - python==3.11.13\n")
                else:
                    f.write(line)

def install_dependencies():
    """Clone and install MintPy and disp-xr with necessary adjustments."""
    # Detect if running in Google Colab
    is_colab = False
    try:
        import google.colab
        is_colab = True
    except ImportError:
        pass

    if not os.path.exists("MintPy"):
        run("git clone https://github.com/insarlab/MintPy.git")
    run("pip install -e ./MintPy")

    if not os.path.exists("disp-xr"):
        run("git clone https://github.com/opera-adt/disp-xr.git")
        
    patch_version_constraints()
    
    run("pip install -e ./disp-xr")
    run("pip install rasterio rioxarray asf_search opera_utils numcodecs s3fs dem_stitcher tile_mate contextily folium zarr h5netcdf h5py")

    # Add source paths for imports
    if is_colab:
        sys.path.insert(0, "/content/disp-xr/src")
        sys.path.insert(0, "/content/MintPy/src")
    else:
        sys.path.insert(0, "disp-xr/src")
        sys.path.insert(0, "MintPy/src")

    print("All dependencies installed successfully.")
