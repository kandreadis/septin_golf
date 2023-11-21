"""
Main script to be run. Please consult the README about the different analysis_type options!
Author: Konstantinos Andreadis
"""
from scripts.analysis import run

if __name__ == "__main__":
    run(analysis_type="cross_section")
    run(analysis_type="surface")
    run(analysis_type="visualise_batch_results")
