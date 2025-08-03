'''
Load all scripts from the etl, analysis, and vis directories.
This script serves as the main entry point for the ETL and analysis pipeline.
It imports functions from the respective modules and executes them in sequence.
'''

import etl.extract as etl_part1
import etl.transform as etl_part2
import etl.load as etl_part3
import analysis.model as analysis_part1
import analysis.evaluate as analysis_part2
import vis.visualizations as visualize

def main():
    # Extract data
    etl_part1.extract_data()

    # Transform data
    etl_part2.transform_data()

    # Load data
    etl_part3.load_data()

    # Analyze data
    analysis_part1.build_model()

    # Evaluate model
    analysis_part2.evaluate_model()

    # Visualize results
    visualize.create_visualizations()

if __name__ == "__main__":
    main()
