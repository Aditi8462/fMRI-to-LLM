# inst414-final-project-Aditi-Kulkarni
Final project for INST414

Project Overview:
    - Business Problem: Can we use machine learning models to detect mental and cognitive states (behavioral responses) from preprocessed fMRI brain imaging data?
    - Datasets Used:
        Dataset link: https://www.openfmri.org/dataset/ds000011/
        Training on classifying objects and counting tone, 4 different tasks:
            - Tone counting: "A task in which a participant needs to count and remember the number of specific tones presented in an experimental run."

            - Single-task weather prediction: "A feedback driven classification learning task in which a subject is presented with a stimuli (ex-geometric shapes) and has to classify them into one of two categories (ex-rainy or sunny weather), and then receives feedback on if the response was correct or incorrect. This may or may not be presented with other stimuli (ex-tones of different frequencies), but the goal of the task is to only pay attention the the classification task at hand."

            - Dual-task weather prediction: "A task in which a subject must attend and respond to two different tasks contained in one experimental run; one task is a feedback driven classification learning task in which a subject is presented with a stimuli (ex-geometric shapes) and has to classify them into one of two categories (ex-rainy or sunny weather), and then receives feedback on whether the response was correct or incorrect. The other task requires the subject to listen to different tones and count the number of a specific tone."

            - Classification probe without feedback: Similar to classification learning task, "preformed after receiving training in the classification learning task. It is similar to the classification learning task but in this task the subject does not receive feedback."

        For Part 2 of the Final Project, I focused on the 'Classification probe without feedback' task, specifically the fMRI BOLD signal data and it's corresponding events.tsv outlining duration and type of trials throughout the task for participant 1 (sub-01)

    - Techniques:
        - Nilearn:
            - load fMRI BOLD signal data
            - Preprocess/mask data to fit standard for predictive models using NiftiMasker
        - scikit-learn:
            - To apply Decision Tree Classifier model
            - To use train_test_split for predictions
            - accuracy_score, recall_score, precision_scroll to evaluate metrics of model
            - plot_tree() to visualize Decision Tree
        - Matplotlib.pyplot:
            - To plot preprocessed results (formated for z-score normalization)
        - Joblib:
            - To save the trained Decision Tree Classifier model for reproducibility through the data science pipeline
        - Seaborn:
            - To plot additional visualizations that help explain the business problem
        - Logging: 
            - For error handling
    
    - Expected Outputs:
        - \data\extracted: 
            - Extracted events.tsv (trial information) and nii.gz (fMRI signal) (nii.gz too big - not in github)
        - \data\processed:
            - task_correlation.csv - correlation values for voxels and the selected trial task
            - mean_bold.csv - mean BOLD signal over the entire scan
            - mean_bold_per_trial.csv - mean BOLD signal per voxel for each trial (for visualization)
            - X.npy - 2D Numpy array of voxel vs time
            - y.csv - labels of trial type matching the X.npy rows
            - Preprocessed fMRI BOLD signal data (npy) and (.nii.gz) file (too big - not in github)
            - labels.csv (trial type)
        - \data\outputs: 
            - test_predictions.csv - test predictions from decision tree model
            - evaluation_metrics.csv - evaluation metrics (accuracy, precision, recall) - how accurate the fMRI BOLD signal is to its classification
            - decision_tree_plot - Decision tree model
            - Mean signal over time of preprocessed fMRI data
            - confusion_matrix.png - how well the model predicts different classes
            - Mean BOLD brain map with location of major activity
            - Mean BOLD signal categorized to trial type 

Setup: 
    - Clone repository on VS Code, and input github HTTPS link to search bar to clone. 
    - Setup Virtual Environment and activate for project using: 
        1. python3 -m venv [environment name] 
        2. source [environment name]/bin/activate
    - Dependencies required and how to install them:
        - nilearn: pip install nilearn (in terminal)
        - pandas: pip install pandas
        - numpy: pip install numpy
        - matplotlib.pyplot: pip install matplotlib
        - os: Available by default (NO installation needed)
        - sickit-learn (sklearn): pip install scikit-learn
        - joblib: pip install joblib
        - Seaborn: pip install seaborn
        - logging: pip install logging

Running the Project:
    - Run main.py, it will automatically download the data in a new directory created in "extract.py" script. The pipeline is fully automated
    - Observe outputs in \data folders

Code Package Structure:
    - \analysis
        - evaluate.py
        - model.py
    - \data
        - \extracted
        - \outputs
        - \processed
        - \reference-tables
    - \elt
        - extract.py
        - load.py
        - transform.py
    - \vis
        - visualizations.py
    - .gitignore 
    - main.py
    - README.md
    - requirements.txt
