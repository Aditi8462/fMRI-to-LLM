"""
Load all scripts from the etl, analysis, and vis directories.
This script serves as the main entry point for the ETL and analysis pipeline.
It imports functions from the respective modules and executes them in sequence.

Each step us logged to a log file named 'pipeline.log' in the main directory.
"""

import os
import logging
import etl.extract as etl_part1
import etl.transform as etl_part2
import etl.load as etl_part3
import analysis.model as analysis_part1
import analysis.evaluate as analysis_part2
import vis.visualizations as visualize

def main():

    #logging inside main()

    base_dir = os.path.dirname(os.path.abspath(__file__)) #main.py directory
    log_file = os.path.join(base_dir, "pipeline.log") #log file path

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
        force = True  # Force logging configuration to be applied, useful for re-running the script without restarting the interpreter
    )


    #try/except statements and add to log file
    try: 
        logging.info("Pipeline started for subject: sub-01, task: Classification Probe without Feedback ")

    # Extract data
        try:
            fMRI_img, events, nii_path, events_path = etl_part1.extract_data() #run this and add to log file with these parameters
            logging.info("Data extracted successfully") #message
        except Exception as e: 
            logging.error(f"Extract step failed: {e}")
            raise

    # Transform data
        try: 
            X_filtered, y_filtered = etl_part2.transform_data()
            logging.info("Data transformed successfully")
        except Exception as e: 
            logging.error(f"Transform step failed: {e}")
            raise

    # Load data
        try:
            X, y = etl_part3.load_data()
            logging.info("Data loaded successfully")
        except Exception as e:
            logging.error(f'Load step failed: {e}')
            raise

    # Analyze data
        try:
            clf, X_test, y_test, predictions = analysis_part1.build_model()
            logging.info("Data model trained successfully")
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise

    # Evaluate model
        try:
            analysis_part2.evaluate_model()
            logging.info("Data model evaluated successfully")
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            raise

    # Visualize results
        try: 
            visualize.create_visualizations()
            logging.info("Visualization created successfully")
        except Exception as e:
            logging.error(f"Visualization failed: {e}")
            raise

        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.critical(f"Pipeline terminated due to errors: {e}")

if __name__ == "__main__":
    main()