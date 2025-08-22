'''
Loads transformed fMRI image data and original events.tsv tabular data.
Creates a new pandas series that defines all the labels for each trial in the duration of task-1.

Run main.py to execute this script.
'''
import pandas as pd
import numpy as np
import os
import logging

def load_data():
    """
    Load the filtered voxel vs. time array and labels CSV made in transform.py.
    """
    processed_path = "data/processed"

    X_file = os.path.join(processed_path, "sub-01_task-Classificationprobewithoutfeedback_X.npy")
    y_file = os.path.join(processed_path, "sub-01_task-Classificationprobewithoutfeedback_y.csv")
    
    # Load filtered data
    X = np.load(X_file)
    logging.info(f"Loaded X file with shape: {X.shape}")

    y = pd.read_csv(y_file)["label"].values
    logging.info(f"Loaded y file with {len(y)} labels")

    return X, y