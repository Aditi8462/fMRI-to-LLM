"""
Visualize the preprocessed fMRI time series and model evaluation outputs.

To visualize preprocessed model: 
    - Used matplotlib to generate average signal over time 
To visualize DecisionTreeClassifer mode: 
    - Used plot_tree to plot model from trained data (kept random_state = 42 so it stays consistent throughout)
To visualize model evaluation outputs:
    - Confusion Matrix to assess model
To visualize transform outputs:
    - A bar plot of mean signal from each trial type
To visuliaze brain signal segregation:
    - Brain map visualization 

Run main.py to execute this script.
"""
#Load necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.tree import plot_tree
import joblib
from nilearn import image, plotting

def create_visualizations():
    """
    Create visualizations for fMRI data and model evaluation.
    - Normalized Confusion Matrix
    - Mean signal over time plot
    - Decision Tree visualization
    - Bar plot of mean signal per trial type
    - Brain map visualization
    """
    # Load time series (already saved as numpy array in data/processed)
    X = np.load('data/processed/sub-01_task-Classificationprobewithoutfeedback_X.npy')  # filtered voxel_vs_time
    y = pd.read_csv("data/processed/sub-01_task-Classificationprobewithoutfeedback_y.csv") #Trial labels as CSV (for visualization)

    # Plot average signal over time (linear model)
    mean_signal_over_time = X.mean(axis=0)  # mean across voxels for each timepoint
    plt.figure(figsize=(12,6))
    plt.plot(mean_signal_over_time)
    plt.title("Mean fMRI Signal Over Time")
    plt.xlabel("Timepoint") #Each timepoint is equal to the RepetitionTime, so each timepoint is 2.0 seconds for 'Classification probe without feedback'
    plt.ylabel("Mean Signal") #This is z-score normailized, so 0 is the mean and values are staggered by 1 standard deviation from the mean
    plt.savefig("data/outputs/mean_signal_over_time.png")
    plt.close()
    logging.info("Saved mean signal over time plot.")

    # Load and display evaluation metrics
    metrics_path = "data/outputs/evaluation_metrics.csv"
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
        print(f"Evaluation Metrics: {metrics}")
    else:
        logging.warning(f"Metrics file not found at {metrics_path}")

    #Making Decision Tree model as visualization, load voxel_vs_time and labels as X and Y
    #Trained Decision Tree for the main model: 
    clf = joblib.load("data/outputs/decision_tree_model.joblib")

    #Visualize Decision tree model using plot_tree()
    plt.figure(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=[f"v{i}" for i in range(X.shape[1])], class_names=list(y["label"].unique()))
    plt.title("Decision Tree Classifier")
    plt.savefig("data/outputs/decision_tree_plot.png")
    plt.close()
    logging.info("Saved Decision Tree plot")

    #-----------------------------Confusion Matrix, Tidy CSV plot, and brain map visualization-----------------------------:

    #Confusion matrix to visualize how well the model fits
    preds_df_path = "data/outputs/test_predictions.csv"
    if os.path.exists(preds_df_path):
        preds_df = pd.read_csv(preds_df_path)
        y_true = preds_df["y_true"]
        y_pred = preds_df["y_pred"]
        cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], normalize='index')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
        plt.title("Confusion Matrix (Normalized)")
        plt.savefig("data/outputs/confusion_matrix.png")
        plt.close()
        logging.info("Saved confusion matrix heatmap.")
    else:
        logging.warning(f"Test predictions file not found at {preds_df_path}")

    #Tidy CSV to plot barplot of trial_type and mean signal for each (from aggregated csv from transform.py)
    tidy_trial_csv = 'data/processed/sub-01_task-Classificationprobewithoutfeedback_mean_bold_per_trial.csv'  # From transform.py
    if os.path.exists(tidy_trial_csv):
        tidy_trial_df = pd.read_csv(tidy_trial_csv)
        plt.figure(figsize=(8, 6))
        sns.barplot(x='trial_type', y='mean_bold', data=tidy_trial_df)
        plt.title("Mean BOLD signal per Trial")
        plt.xlabel("Trial Type")
        plt.ylabel("Mean BOLD signal")
        plt.savefig("data/outputs/mean_bold_per_voxel.png")
        plt.close()
        logging.info("Saved tidy CSV plot for mean BOLD signals.")
    else: 
        logging.warning(f"Trial-type CSV not found at {tidy_trial_csv}")

    #Brain map visualization using plot_stat_map:
    preprocessed_img_path = 'data/processed/sub-01_task-Classificationprobewithoutfeedback_preprocessed.nii.gz'
    if os.path.exists(preprocessed_img_path):
        mean_img = image.mean_img(preprocessed_img_path)
        plotting.plot_stat_map(mean_img, title="Mean BOLD Activity", output_file="data/outputs/mean_bold_brain_map.png")
        plt.close()
        logging.info("Saved brain map visualization.")
    else:
        logging.warning(f"Preprocessed NIfTI not found at {preprocessed_img_path}")

