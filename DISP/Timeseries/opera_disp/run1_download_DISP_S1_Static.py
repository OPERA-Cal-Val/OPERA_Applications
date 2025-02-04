#!/usr/bin/env python3
import argparse
import os, json
from datetime import datetime
from pathlib import Path

from datetime import datetime as dt

import asf_search as asf
from opera_utils.geometry import stitch_geometry_layers
from opera_utils.download import L2Product

from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import botocore

import requests
from io import BytesIO
import zipfile

import warnings
warnings.filterwarnings("ignore")

def createParser(iargs = None):
    '''Commandline input parser'''
    parser = argparse.ArgumentParser(description='Downloading OPERA DISP-S1 from AWS S3 bucket and static layer files from ASF')
    parser.add_argument("--frameID", 
                        required=True, type=str, help='frameID of DISP-S1 to download (e.g., 33039)')
    parser.add_argument("--version",
                        default=0.9, type=float, help='version of DISP-S1 (default: 0.9)') 
    parser.add_argument("--dispDir",
                        default='outputs', type=str, help='directory to download DISP-S1 (default: outputs)')
    parser.add_argument("--startDate", 
                        default='20160101', type=str, help='start date of DISP-S1 (default: 20160101)')
    parser.add_argument("--endDate", 
                        default=dt.today().strftime('%Y%m%d'), type=str, help='end date of DISP-S1 (default: today)')
    parser.add_argument("--nWorkers",
                        default=5, type=int, help='number of simultaenous downloads from AWS S3 bucket (default: 5)')
    parser.add_argument("--staticDir",
                        default='static_lyrs', type=str, help='directory to store static layer files (default: static_lyrs)')
    parser.add_argument("--geomDir",
                        default='geometry', type=str, help='directory to store geometry files from static layers (default: geometry)')
    parser.add_argument("--burstDB-version", 
                        default='0.7.0', type=str, help='burst DB version (default: 0.7.0)')
    parser.add_argument("--staticOnly",
                        action='store_true', help='download only static layer files without nc files')
    return parser.parse_args(args=iargs)

def download_file(bucket_name, file_key, local_path):
    ''' download files from S3 bucket '''
    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED, connect_timeout=600, read_timeout=600, retries={'max_attempts': 10, 'mode': 'standard'}))
    if not os.path.exists(local_path):
        s3.download_file(bucket_name, file_key, local_path)
        print(f"File downloaded to {local_path}")
    else:
        print(f'{local_path} already exists')
    
def list_s3_directories(bucket_name, directory_name, keyword1=None, keyword2=None):
    ''' listing directories in bucket '''
    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))
    paginator = s3.get_paginator('list_objects_v2')
    prefix = directory_name if directory_name.endswith('/') else directory_name + '/'
    directories = set()
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/'):
        for prefix in page.get('CommonPrefixes', []):
            dir_name = prefix['Prefix']
            if (keyword1 is None or keyword1.lower() in dir_name.lower()) and \
               (keyword2 is None or keyword2.lower() in dir_name.lower()):
                directories.add(dir_name)
    return sorted(directories)


def list_s3_files(bucket_name, directory_name):
    '''Listing files in an S3 bucket directory'''
    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))
    paginator = s3.get_paginator('list_objects_v2')
    prefix = directory_name if directory_name.endswith('/') else directory_name + '/'
    files = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            file_name = obj['Key']
            if file_name.endswith('.nc'):  # Ensure we get only .nc files
                files.append(file_name.split('/')[-1])  # Extract only the file name

    return sorted(files)

def get_key(s):
    return '_'.join(s.split('_')[:-2])  # finding last two separated by underscores

def parse_date(date_string):
    return datetime.strptime(date_string, "%Y%m%d")

def filter_list_by_date_range(list_, start_date, end_date):
    ''' filtered based on start and end date '''

    start = parse_date(start_date)
    end = parse_date(end_date)
    
    filtered_list = []
    for item in list_:
        item_start = parse_date(item[30:38])
        item_end = parse_date(item[47:55])

        if (start <= item_start <= end) and (start <= item_end <= end):
            filtered_list.append(item)
    return filtered_list

def main(inps):
    frameID = inps.frameID
    frameID = frameID.zfill(5)    # force frameID to have 5 digit number as string
    version = inps.version 
    dispDir = inps.dispDir
    os.makedirs(dispDir, exist_ok='True')
    startDate = inps.startDate
    endDate = inps.endDate
    nWorkers = inps.nWorkers
    staticDir = inps.staticDir
    os.makedirs(staticDir, exist_ok='True')
    geomDir = inps.geomDir
    os.makedirs(geomDir, exist_ok='True')
    DB_ver = inps.burstDB_version

    # Only process nc files if staticOnly is False
    if not inps.staticOnly:
        bucket_name = 'opera-pst-rs-pop1'       # aws S3 bucket of PST
        #if inps.version == 0.9:
        #    bucket_name = 'opera-int-rs-pop1'

        print('S3 bucket name: ', bucket_name)
        keyword1 = 'F' + frameID
        keyword2 = '_v' + str(version)
        keyword3 = 'v' + str(version).split('.')[1]
        directory_name = f'products/DISP_S1' + '/' + keyword3 + '/' + keyword1  # directory name where DISP-S1s locate
        print('DISP_S1 directory name in bucket: ', directory_name)

        subdirectories = list_s3_directories(bucket_name, directory_name, keyword1=keyword1, keyword2=keyword2)  # search by frame ID
        list_disp = [ dir.split('/')[-2] for dir in subdirectories]
        list_disp = sorted(list_disp)

        unique_dict = {get_key(x): x for x in list_disp}
        list_disp = list(unique_dict.values())

        if not list_disp:  # If no directories found, list files instead
            print("No directories found. Listing files directly...")
            direct_file_mode = True  # Flag to indicate direct file storage
            list_disp = list_s3_files(bucket_name, f'products/DISP_S1/{keyword3}/{keyword1}')
        else:
            direct_file_mode = False

        list_disp = filter_list_by_date_range(list_disp, startDate, endDate)       # filter by dates

        print('Number of DISP-S1 to download:', len(list_disp))
        print('OPERA DISP-S1 data dowload started...')
        # Prepare file paths before the ThreadPoolExecutor
        file_mappings = {
            select_disp: (
                f'products/DISP_S1/{keyword3}/{keyword1}/{select_disp}/{select_disp}.nc',
                f'{dispDir}/{select_disp}.nc'
            ) if not direct_file_mode else (
                f'products/DISP_S1/{keyword3}/{keyword1}/{select_disp}',
                f'{dispDir}/{select_disp}'
            )
            for select_disp in list_disp
        }

        # Concurrent downloading of DISP-S1 nc files
        with ThreadPoolExecutor(max_workers=nWorkers) as executor:
            future_to_file = {
                executor.submit(download_file, bucket_name, file_path, local_path): select_disp
                for select_disp, (file_path, local_path) in file_mappings.items()
            }

            for future in as_completed(future_to_file):
                select_disp = future_to_file[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f'{select_disp} generated an exception: {exc}')

    print('OPERA DISP-S1 data downloaded, moving to static layers... ')
    # Access json matching bursts to frame IDs without downloading
    repo_zip_url = f'https://github.com/opera-adt/burst_db/releases/download/v{DB_ver}/opera-s1-disp-{DB_ver}-frame-to-burst.json.zip'

    # Access the ZIP file
    response = requests.get(repo_zip_url)
    zip_data = BytesIO(response.content)

    # Extract the JSON file from the ZIP archive
    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        json_data = zip_ref.read(f'opera-s1-disp-{DB_ver}-frame-to-burst.json') 

    # Load the JSON data
    data = json.loads(json_data.decode('utf-8'))
    burst_ids = data['data'][frameID.lstrip('0')]['burst_ids']  # list of burst IDs within one frame ID

    # search CLSC Static Layer files
    product = L2Product.CSLC_STATIC

    results = asf.search(
        operaBurstID=list(burst_ids),
        processingLevel=product.value,
    )

    results.download(path=staticDir, processes=5)    # downloading static layers with simultaneous downloads

    list_static_files = [ Path(f'{staticDir}/{results[ii].properties["fileName"]}') for ii in range(len(results)) ] 

    print('number of static layer files to download: ', len(results))
    print(list_static_files)

    # generating los_east.tif and los_north.tif from downloaded static layers
    output_files = stitch_geometry_layers(list_static_files, output_dir=geomDir)

    print('Done')

if __name__ == '__main__':
    # load arguments from command line
    inps = createParser()

    print("==================================================================")
    print("        Downloading DISP-S1 and static layer files")
    print("==================================================================")
    
    # Run the main function
    main(inps)
