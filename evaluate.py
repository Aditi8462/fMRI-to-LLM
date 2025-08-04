"""
Evaluate the trained Decision Tree model using accuracy, precision, and recall.
Save prediction results to data/outputs
Run main.py to execute this script.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model():
    # Load predictions from model.py
    preds_df = pd.read_csv("data/outputs/test_predictions.csv")

    y_true = preds_df["y_true"]
    y_pred = preds_df["y_pred"]

    # Evaluation metrics - accuracy, precision, recall
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # define metrics and save them to a csv file
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    pd.DataFrame([metrics]).to_csv("data/outputs/evaluation_metrics.csv", index=False)
