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

Part 3 Additions: 
    - Saves X (voxel vs time) and y (labels) for analysis and evaluation
    - Compute mean BOLD signal per voxel for reference
    - Compute correlations between mean BOLD and trial types
    - Aggregate the mean BOLD signal per trial type (important to answer business problem)
    - Filters out timepoints where the participant was not doing a task (rest) to prevent overfitting
    
Run main.py to execute this script
'''
import os
import pandas as pd
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn import masking
from nilearn.image import load_img
import logging

def transform_data(SUBJECT, TASK, save_csv=True):  # Aligns with main.py so everything is logged in pipeline.log file
    try:
        logging.info("Starting data transformation")

        # Ensure processed data folder exists
        processed_path = os.path.join('data', 'processed')
        os.makedirs(processed_path, exist_ok=True)

        # Paths to extracted fMRI data and events TSV
        extracted_path = os.path.join('data', 'extracted')
        events_path = os.path.join(extracted_path, f'{SUBJECT}_{TASK}_events.tsv')
        events = pd.read_csv(events_path, sep='\t')

        fMRI_file = os.path.join(extracted_path, f'{SUBJECT}_{TASK}_bold.nii.gz')
        fMRI_img = load_img(fMRI_file)  # Load the extracted fMRI image

        # Create a brain mask to determine which voxels belong to the brain
        mask_img = masking.compute_epi_mask(fMRI_img)

        # Perform preprocessing on the fMRI data using NiftiMasker
        masker = NiftiMasker(
            mask_img=mask_img,      # Use the brain mask
            standardize=True,       # z-score normalization per voxel
            detrend=True,           # Remove slow linear trends
            smoothing_fwhm=6.0,     # Gaussian smoothing (6mm FWHM)
            high_pass=0.01,         # Filter out very slow changes (<0.01 Hz)
            low_pass=0.1,           # Filter out very fast changes (>0.1 Hz)
            t_r=2.0,                # Repetition time of fMRI acquisition
        )

        # Transform the 4D fMRI image into a 2D array: timepoints x voxels
        voxel_vs_time = masker.fit_transform(fMRI_img)
        logging.info(f"Voxel x Time shape: {voxel_vs_time.shape}")

        # Save the preprocessed 4D image
        preprocessed_img = masker.inverse_transform(voxel_vs_time)
        preprocessed_file = os.path.join(processed_path, f'{SUBJECT}_{TASK}_preprocessed.nii.gz')
        preprocessed_img.to_filename(preprocessed_file)
        logging.info(f"Saved preprocessed NIfTI: {preprocessed_file}")

        # Compute mean BOLD signal per voxel for reference
        mean_signal = voxel_vs_time.mean(axis=1)
        tidy_df = pd.DataFrame({'mean_bold': mean_signal})
        tidy_csv = os.path.join(processed_path, f'{SUBJECT}_{TASK}_mean_bold.csv')
        tidy_df.to_csv(tidy_csv, index=False)
        logging.info(f"Tidy CSV saved: {tidy_csv}")

        # Align timepoints to trials using events.tsv
        n_timepoints = voxel_vs_time.shape[0]
        tr = 2.0
        timepoints_sec = np.arange(n_timepoints) * tr
        timepoint_labels = np.full(n_timepoints, "rest", dtype=object)

        for _, row in events.iterrows():
            onset = row["onset"]
            duration = row["duration"]
            label = row["trial_type"]
            in_event = (timepoints_sec >= onset) & (timepoints_sec < onset + duration)
            timepoint_labels[in_event] = label

        # Filter out "rest" timepoints to ensure X and y match exactly
        trial_mask = timepoint_labels != "rest"
        X_filtered = voxel_vs_time[trial_mask, :]
        y_filtered = timepoint_labels[trial_mask]

        # Save filtered X and y for modeling
        np.save(os.path.join(processed_path, f"{SUBJECT}_{TASK}_X.npy"), X_filtered)
        pd.DataFrame({"label": y_filtered}).to_csv(os.path.join(processed_path, f"{SUBJECT}_{TASK}_y.csv"), index=False)

        # Compute correlations between mean BOLD and trial types
        correlations = {}
        for trial_type in np.unique(timepoint_labels):
            if trial_type == "rest":
                continue
            trial_mask_type = timepoint_labels == trial_type
            correlations[trial_type] = np.corrcoef(mean_signal[trial_mask_type], np.ones(trial_mask_type.sum()))[0, 1]

        correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['correlation'])
        correlation_csv_path = os.path.join(processed_path, f'{SUBJECT}_{TASK}_bold_task_correlation.csv')
        if save_csv:
            correlation_df.to_csv(correlation_csv_path)
        logging.info(f"Correlation CSV saved: {correlation_csv_path}")

        # Aggregate mean BOLD per trial_type
        mean_per_trial = []
        for trial_type in np.unique(timepoint_labels):
            if trial_type == "rest":
                continue
            trial_mask_type = timepoint_labels == trial_type
            mean_val = voxel_vs_time[trial_mask_type, :].mean()
            mean_per_trial.append({'trial_type': trial_type, 'mean_bold': mean_val})

        trial_tidy_df = pd.DataFrame(mean_per_trial)
        trial_tidy_csv = os.path.join(processed_path, f'{SUBJECT}_{TASK}_mean_bold_per_trial.csv')
        trial_tidy_df.to_csv(trial_tidy_csv, index=False)
        logging.info(f"Mean BOLD per trial_type tidy CSV saved: {trial_tidy_csv}")

        return X_filtered, y_filtered

    except Exception as e:
        logging.error(f"Transform step failed: {e}")
        raise
