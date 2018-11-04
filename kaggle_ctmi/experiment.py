"""
A simple solution to CT / CTA detection based in Kaggle datasets.
"""

import argparse
import logging
import os
import time

import numpy as np

from kaggle_ctmi.algorithm import Algorithm
from kaggle_ctmi.cohort import ShaipWorkspace, Cohort
from kaggle_ctmi.results import Results


class Experiment(object):
    """ This is the top-level class, orchestrating train/test split of the cohort,
    training and evaluation.  However he details are all elsewhere"""

    def __init__(self, shaip_root_dir):
        self.shaip = ShaipWorkspace(shaip_root_dir)
        self.shaip.check()
        self.algorithm = Algorithm(self.shaip.cache_dir)
        self.results = Results(self.shaip.results_dir)
        self.args = None

    def command_line(self):
        parser = argparse.ArgumentParser(description='CT/CTA discrimination to run in SHAIP')
        parser.add_argument('-nt', '--notrain', help='skip training step (assumes model exists',
                            action='store_true', default=False)
        self.args = parser.parse_args()

    def setup_logging(self):
        # see https://docs.python.org/2.4/lib/multiple-destinations.html

        logger = logging.getLogger('')
        logger.setLevel(logging.DEBUG)

        if len(logger.handlers) <= 1:
            # avoid double setup which can happen in unit tests

            # Define a Handler which writes INFO messages or higher to the sys.stderr
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            simple_formatter = logging.Formatter('%(levelname)-8s %(message)s')
            console_handler.setFormatter(simple_formatter)

            # Set up logging to file for DEBUG messages or higher
            logfile_path = os.path.join(self.shaip.results_dir, 'kaggle-ctmi.log')
            logfile_handler = logging.FileHandler(filename=logfile_path)
            logfile_handler.setLevel(logging.DEBUG)
            verbose_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%d/%m/%y %H:%M')
            logfile_handler.setFormatter(verbose_formatter)

            # add the handlers to the logger
            logger.addHandler(console_handler)
            logger.addHandler(logfile_handler)

            # Silence matplotlib debug messages
            mpl_logger = logging.getLogger('matplotlib.font_manager')
            mpl_logger.setLevel(logging.WARNING)

    def main(self):
        """ Main Experiment entry point """

        self.setup_logging()
        self.command_line()
        start_time = time.time()

        logging.info("Starting Kaggle-CTMI Experiment\n")

        logging.info("Loading data and groundtruth...")
        cohort = Cohort(self.shaip)
        train_cohort, test_cohort = cohort.split_cohort_train_test(0.3)
        logging.info("Loaded %d datasets", cohort.size)

        if self.args.notrain:
            logging.info("Skipping training, using saved model")
            model = self.algorithm.load_model(self.shaip.models_dir + 'model')
        else:
            logging.info("Training on %d datasets...", train_cohort.size)
            model = self.algorithm.train(train_cohort)
            Algorithm.save_model(model, self.shaip.models_dir + 'model')

        logging.info("Prediction on %d datasets...", test_cohort.size)
        test_predictions = self.algorithm.predict(model, test_cohort)

        logging.info("Generating results to ShaipWorkspace/outputs/results/index.html...")
        self.results.show_results(train_cohort, test_cohort,
                                  self.algorithm.history, test_predictions)

        logging.info("Kaggle-CTMI Experiment done in %4.1f seconds.\n", (time.time() - start_time))


# Lets do it!
if __name__ == '__main__':
    np.random.seed(42)

    expt = Experiment('ShaipWorkspace/')
    expt.main()
