'''
Make a DecisionTreeClassifer model using voxels and timepoints as X-value and labels as Y-value.
Save model to data/outputs
Run main.py to execute this script.
'''
#Load necessary libraries:
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def build_model():
    # Load features and labels
    voxel_vs_time = np.load('data/processed/sub-01_task-1.npy')
    labels = pd.read_csv("data/processed/sub-01_task-1_labels.csv")["label"]

    # Split into train and test sets (test = 20%, train = 80%)
    X_train, X_test, y_train, y_test = train_test_split(
        voxel_vs_time, 
        labels, 
        test_size=0.2,
        random_state = 42
    )

    # Train decision tree classifier with maxdepth of 5
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # Save predictions and model
    predictions = clf.predict(X_test)
    pd.DataFrame({"y_true": y_test, 
                  "y_pred": predictions}).to_csv(
                  "data/outputs/test_predictions.csv", 
                  index=False
    )

    return clf, X_test, y_test, predictions
