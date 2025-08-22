'''
Extract data from flat files downloaded from OpenfMRI.
This script opens one file from sub-01 (participant 1 of the experiment)
There are 4 tasks in the experiment (specified in README.md), and this script extracts data for the first task.

This script extracts the fMRI BOLD signal data and the corresponding events.tsv file (Task and durations).
NOTE: The two datasets are directly stored in data/raw when you run main.py

Imports: 
    - Externally download the files from OpenfMRI using requests.
    - Used os to create a directory to store the extracted data.
    - Used pandas to read the events.tsv file.
    - Used nilearn to load the fMRI BOLD signal image data (NIfTI file).
Outputs:
    - Saves the extracted data in a directory called 'extracted' inside the 'data' folder.
    
Run main.py to execute this script.
'''

# Load necessary libraries
import os
import logging
import pandas as pd
from nilearn.image import load_img
import requests

#------------------------------------Define function to download file--------------------------------------

def download_file(url, dest_path):
    """
    Download a file from a URL to store the extracted data: 
    - fMRI BOLD signal data (NIfTI file)
    - events.tsv file (Task and durations)
    """
    # Check if the file already exists
    if os.path.exists(dest_path):
        logging.info(f"File {dest_path} already exists, skipping download.")
        return dest_path
    
    # if the file does not exist, download it
    logging.info(f"Downloading {url}...")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True) # Create directory if it does not exist

    with requests.get(url, stream=True) as r:
        r.raise_for_status() #Raise error for bad responses
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): #Ensures the response is streamed in chunks to prevemt crashing, chunk size is 8192 bytes
                f.write(chunk)
    logging.info(f"File downloaded to {dest_path}")
    return dest_path

#------------------------------------Function to extract data--------------------------------------

def extract_data():
    """
    Extract fMRI BOLD signal data and corresponding events.tsv file for sub-01, task: Classification Probe without Feedback.
    """
    # Create a directory to store raw data in the data folder 
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    # Create a directory to store extracted data in the data folder
    extracted_dir = os.path.join("data", "extracted")
    os.makedirs(extracted_dir, exist_ok=True)

    #S3 URL for the flat files
    nii_url = (
        f"https://s3.amazonaws.com/openneuro/ds000011/ds000011_R2.0.1/uncompressed/sub-01/func/sub-01_task-Classificationprobewithoutfeedback_bold.nii.gz"
    )

    events_url = (
        f"https://s3.amazonaws.com/openneuro/ds000011/ds000011_R2.0.1/uncompressed/sub-01/func/sub-01_task-Classificationprobewithoutfeedback_events.tsv"
    )

    #Download the files and store them in the raw directory
    nii_file = os.path.join(raw_dir, "sub-01_task-Classificationprobewithoutfeedback_bold.nii.gz")
    events_file = os.path.join(raw_dir, "sub-01_task-Classificationprobewithoutfeedback_events.tsv")

    download_file(nii_url, nii_file)
    download_file(events_url, events_file)

    #------------------------------------Load the fMRI BOLD signals file--------------------------------------

    ## fMRI BOLD data, in nii.gz file
    # Load fMRI BOLD signal image data (NIfTI file) into memory
    fMRI_img = load_img(nii_file)
    nii_path = os.path.join("data/extracted", "sub-01_task-Classificationprobewithoutfeedback_bold.nii.gz")
    # Save the file to the data folder
    fMRI_img.to_filename(nii_path)

    logging.info(f"fMRI image loaded and saved to {nii_path}")

    #------------------------------------Load the trial tasks and duration file--------------------------------------

    ### Task, in events,tsv file
    ## Used sep = '\t' to read the file, as it is in TSV, not CSV
    # Load corresponding events.tsv file (flat file)
    events = pd.read_csv(events_file, sep='\t')

    # Save the file to the data folder
    events_path = os.path.join(extracted_dir, "sub-01_task-Classificationprobewithoutfeedback_events.tsv")
    events.to_csv(events_path, sep='\t', index=False)

    logging.info(f"Events file loaded and saved to {events_path}")

    return fMRI_img, events, nii_path, events_path