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

Run main.py to execute this script.
'''

# Load necessary libraries
import os
import pandas as pd
from nilearn.image import load_img

def extract_data():
    # Create a directory to store extracted data in the data folder
    extracted = 'data/extracted'
    os.makedirs(extracted, exist_ok=True)

    ## fMRI BOLD data, in nii.gz file
    # Load fMRI BOLD signal image data (NIfTI file) into memory
    # File path to the NIfTI file, stored in my local computer
    fMRI_img = load_img('/Users/Aditi/Documents/UMD/AI + NEUR - Chicoli/fMRI-Preprocessing/ds000011_R2.0.1/sub-01/func/sub-01_task-Classificationprobewithoutfeedback_bold.nii.gz')

    # Save the file to the data folder
    fMRI_img.to_filename(os.path.join(extracted, 'sub-01_task-Classificationprobewithoutfeedback_bold.nii.gz'))


    ### Task, in events,tsv file
    ## Used sep = '\t' to read the file, as it is in TSV, not CSV
    # Load corresponding events.tsv file (flat file)
    events = pd.read_csv('/Users/Aditi/Documents/UMD/AI + NEUR - Chicoli/fMRI-Preprocessing/ds000011_R2.0.1/sub-01/func/sub-01_task-Classificationprobewithoutfeedback_events.tsv', sep='\t')

    # Save the file to the data folder
    events.to_csv(os.path.join(extracted, 'sub-01_task-Classificationprobewithoutfeedback_events.tsv'), sep='\t', index=False)

    return fMRI_img, events
