import numpy as np

from src.main.utilities.utils import plot_metrics_histogram

if __name__ == '__main__':
    plot_metrics_histogram("cup_submission", "cup_submission_refinement", "Validation")
    plot_metrics_histogram("cup_submission", "cup_submission_refinement", "Test")
    plot_metrics_histogram("cup_submission", "cup_submission_refinement", "Training")