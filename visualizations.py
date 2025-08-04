"""
Visualize the preprocessed fMRI time series and model evaluation outputs.
To visualize preprocessed model: 
    - Used matplotlib to generate average signal over time 
    **I am still deciding how and if I want to plot the preprocessed results, I think it might be helpful

To visualize DecisionTreeClassifer mode: 
    - Used plot_tree to plot model from trained data (kept random_state = 42 so it stays consistent throughout)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

def create_visualizations():
    # Load time series (already saved as numpy array in data/processed)
    voxel_vs_time = np.load('data/processed/sub-01_task-1.npy')

    # Plot average signal over time (linear model)
    mean_signal = voxel_vs_time.mean(axis=1)
    plt.plot(mean_signal)
    plt.title("Mean fMRI Signal Over Time")
    plt.xlabel("Timepoint") #Each timepoint is equal to the RepetitionTime, so each timepoint is 2.0 seconds for 'Classification probe without feedback'
    plt.ylabel("Mean Signal") #This is z-score normailized, so 0 is the mean and values are staggered by 1 standard deviation from the mean
    plt.savefig("data/outputs/mean_signal_over_time.png")
    plt.close()

    # Load and display evaluation metrics
    metrics = pd.read_csv("data/outputs/evaluation_metrics.csv")
    print("Evaluation metrics:\n", metrics)

    #Making Decision Tree model as visualization, load voxel_vs_time and labels as X and Y
    X = np.load('data/processed/sub-01_task-1.npy')     # voxel_vs_time as numpy array (2D)
    y = pd.read_csv("data/processed/sub-01_task-1_labels.csv") #Trial labels as CSV (for visualization)

    #Train Decision Tree for the main model: 
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    #Visualize Decision tree model using plot_tree()
    plt.figure(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=[f"v{i}" for i in range(X.shape[1])], class_names=True)
    plt.title("Decision Tree Classifier")
    plt.savefig("data/outputs/decision_tree_plot.png")
    plt.close()
