'''
Loads transformed fMRI image data and original events.tsv tabular data.
Creates a new pandas series that defines all the labels for each trial in the duration of task-1.


NOTE: I had to troubleshoot this script because the rows of labels.csv and events.tsv were not equal
(gave me 'inconsistent numbers of samples: [240, 113]" error)

Part 3 Additions: 
    - Simplified this so it loads X and y for analysis and eval, rest of computation is in transform.py

Run main.py to execute this script.
'''
import pandas as pd
import numpy as np
import os

def load_data(SUBJECT="sub-01", TASK="Classificationprobewithoutfeedback"):
    """
    Load the filtered voxel vs. time array and labels CSV made in transform.py.
    Ensures X and y have the same number of samples - I kept having issues with this.
    """
    processed_path = "data/processed"

    X_file = os.path.join(processed_path, f"{SUBJECT}_{TASK}_X.npy")
    y_file = os.path.join(processed_path, f"{SUBJECT}_{TASK}_y.csv")
    
    # Load filtered data
    X = np.load(X_file)
    y = pd.read_csv(y_file)["label"].values

    return X, y
