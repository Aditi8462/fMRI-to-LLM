'''
Extract data from flat files downloaded from OpenfMRI.
This script opens one file from sub-01 (participant 1 of the experiment)
There are 4 tasks in the experiment (specified in README.md), and this script extracts data for the first task.

This script extracts the fMRI BOLD signal data and the corresponding events.tsv file (Task and durations).
NOTE: The two datasets come from a very large file which takes a long time to commit and push up to github, so I have extracted them from my local computer and saved them under data/extracted

Imports: 
    - Used os to create a directory to store the extracted data.
    - Used pandas to read the events.tsv file.
    - Used nilearn to load the fMRI BOLD signal image data (NIfTI file).
Outputs:
    - Saves the extracted data in a directory called 'extracted' inside the 'data' folder.

Part 3 change: 
    - Used a dynamic directory approach to avoid hardcoded values
    
Run main.py to execute this script.
'''

# Load necessary libraries
import os
import pandas as pd
from nilearn.image import load_img

def extract_data(subject = "sub-01", task = "Classificationprobewithoutfeedback"):
    ### Get flat files from /data/raw - Part 3 addition
    raw_dir = os.path.join("data", "raw")
    # Create a directory to store extracted data in the data folder
    extracted = os.path.join("data", "extracted")
    os.makedirs(extracted, exist_ok=True)

    #Placeholder for directory of files
    nii_file = None
    events_file = None

    #ensures it stored the correct file based on the task, into nii_file and events_file
    for file in os.listdir(raw_dir): 
        if file.endswith("nii.gz") and task in file:
            nii_file = os.path.join(raw_dir, file)
        elif file.endswith("_events.tsv") and task in file:
            events_file = os.path.join(raw_dir, file)
    
    if nii_file is None or events_file is None: 
        raise FileNotFoundError(f"Could not find .nii.gz file or events.tsv file for {subject}, task {task}")
    
    ## fMRI BOLD data, in nii.gz file
    # Load fMRI BOLD signal image data (NIfTI file) into memory
    fMRI_img = load_img(nii_file)
    nii_extracted = os.path.join("data/extracted", f"{subject}_{task}_bold.nii.gz")
    # Save the file to the data folder
    fMRI_img.to_filename(nii_extracted)


    ### Task, in events,tsv file
    ## Used sep = '\t' to read the file, as it is in TSV, not CSV
    # Load corresponding events.tsv file (flat file)
    events = pd.read_csv(events_file, sep='\t')

    # Save the file to the data folder
    events_path = os.path.join(extracted, f"{subject}_{task}_events.tsv")
    events.to_csv(events_path, sep='\t', index=False)

    return fMRI_img, events
