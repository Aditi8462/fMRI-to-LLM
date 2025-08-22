'''
Make a DecisionTreeClassifer model using voxels and timepoints as X-value and labels as Y-value.
Save model & test files to data/outputs

Run main.py to execute this script.
'''
#Load necessary libraries:
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib #To save decision tree model for future scripts
import logging

def build_model():
    """
    Build and train a Decision Tree Classifier model.
    """
    #State directory for results to go in (data/outputs)
    outputs_dir = os.path.join("data", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Load features and labels
    processed_path = "data/processed"
    X_file = os.path.join(processed_path, "sub-01_task-Classificationprobewithoutfeedback_X.npy")
    y_file = os.path.join(processed_path, "sub-01_task-Classificationprobewithoutfeedback_y.csv")

    X = np.load(X_file)
    y = pd.read_csv(y_file)["label"].values

    logging.info(f"Loaded filtered data: X shape {X.shape}, y shape {y.shape}")

    # Split into train and test sets (test = 20%, train = 80%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2,
        random_state = 42
    )

    # Logging info
    logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Train decision tree classifier with maxdepth of 5
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # Save trained model using joblib
    model_path = os.path.join(outputs_dir, "decision_tree_model.joblib")
    joblib.dump(clf, model_path)

    # logging info
    logging.info(f"Trained model saved to {model_path}")

    # Save test data:

    np.save(os.path.join(outputs_dir, "X_test.npy"), X_test)
    pd.DataFrame({"label": y_test}).to_csv(os.path.join(outputs_dir, "y_test.csv"), index=False)

    # Save predictions and model
    predictions = clf.predict(X_test)
    pd.DataFrame({"y_true": y_test, 
                  "y_pred": predictions}).to_csv(
                  os.path.join(outputs_dir, "test_predictions.csv"),
                  index=False
    )

    # logging info
    logging.info(f"Predictions shape: {predictions.shape}, Test labels shape: {y_test.shape}")

    return clf, X_test, y_test, predictions