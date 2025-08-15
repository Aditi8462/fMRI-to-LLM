"""
Evaluate the trained Decision Tree model using accuracy, precision, and recall.
Save prediction results to data/outputs

Part 3 Addition: 
    -Saved the trained model using joblib
    
Run main.py to execute this script.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging
import joblib #To load the mdoel instead of retraining it every time

def evaluate_model():

    logging.info("Starting evaluation of model..")
    # Load predictions from model.py
    outputs_dir = os.path.join("data", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    #Load the trained decision tree model using joblib
    model_path = os.path.join(outputs_dir, "decision_tree_model.joblib")
    clf = joblib.load(model_path)
    logging.info(f"Loaded trained model from {model_path}")

    #Load the test data: 
    X_test = np.load(os.path.join(outputs_dir, "X_test.npy"))
    y_test = pd.read_csv(os.path.join(outputs_dir, "y_test.csv"))["label"]

    #Generate the predictions from X_test
    y_pred = clf.predict(X_test)

    #Save predictions in a dataframe
    preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    preds_path = os.path.join(outputs_dir, "test_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    logging.info(f"Predictions saved to {preds_path}")

    # Evaluation metrics - accuracy, precision, recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    # define metrics and save them to a csv file
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    metrics_path = os.path.join(outputs_dir, "evaluation_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logging.info(f"Metrics saved to {metrics_path}")
