# This script downloads DSWx-HLS data that has been generated over Lake Mead between 2014-2023 and stored in a public S3 bucket
from pathlib import Path
import os
import numpy as np
from collections import defaultdict
import multiprocessing

os.environ['AWS_NO_SIGN_REQUEST'] = "YES"

# download helper
def download_one_tile(tile_url):
    band_id = tile_url.split('/')[-1].split('_')[-2]
    wget_command = f"wget  -nc -q -O ../data/{band_id}/{tile_url.split('/')[-1]} {tile_url}"
    os.system(wget_command)

# main function
def main():
    band_ids = ["B01", "B03", "B04"]
    tile_ids = ["T11SPA", "T11SPV", "T11SQA", "T11SQV"]
    aws_prefix = "https://opera-pst-rs-pop1.s3.us-west-2.amazonaws.com"
    file_manifest = Path('../dswx_recursive.txt')

    tile_list = defaultdict(list)
    with open(file_manifest, 'r') as f:
        lines = f.readlines()

    for line in lines:
        entry_name = line.split(" ")[-1].strip()
        dswx_filename = entry_name.split('/')[-1]
        if dswx_filename[-3:] != 'tif':
            continue
        tile_id, band_id = dswx_filename.split("_")[3], dswx_filename.split("_")[-2]

        if np.all([np.in1d(tile_id, tile_ids), np.in1d(band_id, band_ids)]):
            tile_list[band_id].append(f"{aws_prefix}/{entry_name}")

    for key in tile_list.keys():
        tile_list[key] = sorted(tile_list[key])
        download_path = Path(f'../data/{key}')
        if not download_path.exists():
            download_path.mkdir()

    tile_urls = []
    for value in tile_list.values():
        tile_urls.extend(value)

    # Create a multiprocessing Pool with the number of desired processes
    num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    # Use the pool to map the calculate function to the inputs in parallel
    _ = pool.map(download_one_tile, tile_urls)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
