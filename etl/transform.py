'''
Transform raw fMRI data into a preprocessed format for analysis.
Uses NiftiMasker to preprocess the data

Imports:
    - Used os to create a directory to store the processed data.
    - Used nilearn to apply NiftiMasker for preprocessing on extracted data
    - Used nilearn to load the fMRI BOLD signal image data (NIfTI file).
Outputs: 
    - Saves preprocessed fMRI BOLD signal data (as a 2D Numpy array)
    - Saves preprocessd fMRI BOLD signal data in original format (4D spatial data)

Run main.py to execute this script
'''
import os
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn import image, masking
from nilearn.image import load_img

def transform_data(): 

    processed = 'data/processed'
    os.makedirs(processed, exist_ok=True)

    # Load the extracted fMRI image data
    fMRI_img = load_img('data/extracted/sub-01_task-Classificationprobewithoutfeedback_bold.nii.gz')
    
    #Create a brain mask from the fMRI image data (to detemine which voxels are part of the brain (1) and which are not (0))
    mask_img = masking.compute_epi_mask(fMRI_img)

    #Perform preprocessing on the fMRI data using NiftiMasker
    masker = NiftiMasker(
        mask_img=mask_img,      # Use the brain mask created above (binary data)
        standardize=True,       # Standardize the data (mean=0, std=1) to make it easier to compare across different voxels (z-score normalization)) - this way, each voxel is measured in the same way
        detrend=True,           # Removes slow linear trends in data - voxels' time series can increase or decrease over time, so this removes that trend because it's not related to neural activity
        smoothing_fwhm=6.0,     # Gaussian smoothing with a full-width at half maximum of 6mm - this blurs the image slightly so the signal of each voxel is averaged with its neigbors
        high_pass=0.01,         # Filters out frequencies below 0.01 Hz (slow changes in signal (like physical shifts) that interfere with brain response signals)
        low_pass=0.1,           # Filters out frequencies above 0.1 Hz (fast changes in signal (like heartbeat noises) that interfere with brain response signals)
        t_r=2.0,                # Repetition time for task, found under: ds000011_R2.0.1/task-Classificationprobewithoutfeedback_bold.json
    )
    ## Fit the masker to the fMRI image data and transform it into a time series 
    # This will return a 2D array where each row is a voxel and each column is a time point - no spatial info, so it can be saved
    # And it can also have analytics performed on it
    voxel_vs_time = masker.fit_transform(fMRI_img)

    ## To save original preprocessed image
    # Covert back to 4D array with spatial info, and save to data/processed folder
    preprocessed_img = masker.inverse_transform(voxel_vs_time)

    preprocessed_img.to_filename(os.path.join(processed, 'sub-01_task-Classificationprobewithoutfeedback_preprocessed.nii.gz'))

    # Save 2D numpy array to processed folder as well
    np.save('data/processed/sub-01_task-1.npy', voxel_vs_time)

    return voxel_vs_time