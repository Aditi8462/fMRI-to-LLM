'''
Loads transformed fMRI image data and original events.tsv tabular data.
Creates a new pandas series that defines all the labels for each trial in the duration of task-1.
Run main.py to execute this script.

NOTE: I had to troubleshoot this script because the rows of labels.csv and events.tsv were not equal
(gave me 'inconsistent numbers of samples: [240, 113]" error)
'''
import pandas as pd
import numpy as np
from nilearn.image import load_img

def load_data():
    # Load the preprocessed 4D image
    preprocessed_img = load_img('data/processed/sub-01_task-Classificationprobewithoutfeedback_preprocessed.nii.gz')
    
    # Load the 2D numpy array (for future analysis and evaluation)
    voxel_vs_time = np.load('data/processed/sub-01_task-1.npy')
    
    # Load events file for labels
    events = pd.read_csv('data/extracted/sub-01_task-Classificationprobewithoutfeedback_events.tsv', sep='\t')

    ## Extract labels (target variables for decisiontreeclassifier in the future), and save to \data\processed

    #Issue withe label values, so initializing labels so events.tsv and labels.csv rows will match
    n_timepoints = voxel_vs_time.shape[0]
    tr = 2.0  # RepetitionTime in seconds
    timepoints_sec = np.arange(n_timepoints) * tr

    # Initialize all labels as "rest" - this fills the rest of the values out so number of rows match
    timepoint_labels = np.full(n_timepoints, "rest", dtype=object)

    # Assigned trial label to each timepoint if it falls within any event (row)
    for _, row in events.iterrows():
        onset = row["onset"]
        duration = row["duration"]
        label = row["trial_type"]

        in_event = (timepoints_sec >= onset) & (timepoints_sec < onset + duration)
        timepoint_labels[in_event] = label

    # Convert 2D numpy to CSV format to visualize on VS Code
    # Gave this a header named "label" for future parts 
    pd.DataFrame({"label": timepoint_labels}).to_csv("data/processed/sub-01_task-1_labels.csv", index=False)

    return preprocessed_img, voxel_vs_time, timepoint_labels