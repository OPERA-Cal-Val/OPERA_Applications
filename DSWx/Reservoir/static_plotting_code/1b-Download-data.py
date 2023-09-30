# %%
# A separate file was provided for DSWx-HLS tiles over Lake Mead generated for the date range 06/22-04/23
# Those files are downloaded in this script.

from pathlib import Path
import os
import multiprocessing

os.environ['AWS_NO_SIGN_REQUEST'] = "YES"

# download helper
def download_one_tile(tile_url):
    band_id = tile_url.split('/')[-1].split('_')[-2]
    wget_command = f"wget  -nc -q -O ../data/{band_id}/{tile_url.split('/')[-1]} {tile_url}"
    os.system(wget_command)

# main function
def main():
    suffixes = {"B01": "_B01_WTR.tif", "B03": "_B03_CONF.tif", "B04": "_B04_DIAG.tif"}
    aws_prefix = "https://opera-pst-rs-pop1.s3.us-west-2.amazonaws.com/"
    file_manifest = Path('../lake_mead_dswx_manifest.txt')

    with open(file_manifest, 'r') as f:
        lines = f.readlines()

    tile_urls = []
    for line in lines:
        entry_name = line.strip()
        dswx_filename = entry_name.split('/')[-2]
        prefix = aws_prefix + '/'.join(entry_name.split('/')[3:6])+'/'
        for _, value in suffixes.items():
            this_file = prefix + dswx_filename + value
            tile_urls.append(this_file)

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