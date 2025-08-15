'''
Load all scripts from the etl, analysis, and vis directories.
This script serves as the main entry point for the ETL and analysis pipeline.
It imports functions from the respective modules and executes them in sequence.

Part 3 Addition: After changing all scripts, I added logging to run through each part of the code and show the log in pipeline.log
'''
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
    os.makedirs('data/outputs', exist_ok=True)
    logging.basicConfig(
        filename='data/outputs/pipeline.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    #Define a dynamic folder for flat file loading
    DATA_ROOT = os.path.join('data', 'raw') #Defines the folder
    SUBJECT = 'sub-01' #Only looking at one subject for this project
    TASK = 'Classificationprobewithoutfeedback' #The task we are looking at

    #try/except statements and add to log file
    try: 
        logging.info("Pipeline started")

    # Extract data
        try:
            fmri_img, events = etl_part1.extract_data(SUBJECT, TASK) #run this and add to log file with these parameters
            logging.info("Data extracted successfully") #message
        except Exception as e: 
            logging.error(f"Extract step failed: {e}")
            raise

    # Transform data
        try: 
            X_filtered, y_filtered = etl_part2.transform_data(SUBJECT, TASK)
            logging.info("Data transformed successfully")
        except Exception as e: 
            logging.error(f"Transform step failed: {e}")
            raise

    # Load data
        try:
            X, y = etl_part3.load_data(SUBJECT, TASK)
            logging.info("Data loaded successfully")
        except Exception as e:
            logging.error(f'Load step failed: {e}')
            raise

    # Analyze data
        try:
            clf, X_test, y_test, predictions = analysis_part1.build_model(SUBJECT, TASK)
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
