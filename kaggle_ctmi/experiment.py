"""
A simple solution to CT / CTA detection based in Kaggle datasets.
"""

import argparse
import logging
import os
import sys
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

    def command_line(self, argv):
        parser = argparse.ArgumentParser(
            prog='experiment.py',
            description='CT/CTA discrimination to run in SHAIP',
            epilog='If no phases are specified, program does nothing - exits')
        parser.add_argument('-t', '--train', help='perform model training',
                            action='store_true', default=False)
        parser.add_argument('-p', '--predict', help='perform prediction over the test set',
                            action='store_true', default=False)
        parser.add_argument('-e', '--evaluate', help='generate results',
                            action='store_true', default=False)

        args = parser.parse_args(argv[1:])
        if not any([args.train, args.predict, args.evaluate]):
            parser.print_help()
            sys.exit(0)
        self.args = args

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

    def main(self, argv):
        """ Main Experiment entry point.
        argv is the full argument list, so argv[0] is the program name.  In production
        call as main(sys.argv)"""

        np.random.seed(42)
        self.setup_logging()
        self.command_line(argv)
        start_time = time.time()

        logging.info("Starting Kaggle-CTMI Experiment\n")

        logging.info("Finding data and groundtruth...")
        cohort = Cohort(self.shaip)
        train_cohort, test_cohort = cohort.split_cohort_train_test(0.3)
        logging.info("Found %d datasets", cohort.size)

        if self.args.train:
            logging.info("Training on %d datasets...", train_cohort.size)
            model = self.algorithm.train(train_cohort)
            Algorithm.save_model(model, self.shaip.models_dir + 'model')
        else:
            logging.info("Skipping training, model saved from earlier run")
            model = self.algorithm.load_model(self.shaip.models_dir + 'model')

        if self.args.predict:
            logging.info("Prediction on %d datasets...", test_cohort.size)
            test_predictions = self.algorithm.predict(model, test_cohort)
        else:
            logging.info("Skipping prediction, using predictions from earlier run")
            # TODO: need to sort out caching of predictions
            test_predictions = None

        if self.args.evaluate:
            logging.info("Generating results to ShaipWorkspace/outputs/results/index.html...")
            self.results.show_results(train_cohort, test_cohort,
                                      self.algorithm.history, test_predictions)

        logging.info("Kaggle-CTMI Experiment done in %4.1f seconds.\n", (time.time() - start_time))


# Lets do it!
if __name__ == '__main__':
    expt = Experiment('ShaipWorkspace/')
    # You'll need to set command arguments - e.g. --train --predict --evaluate (or -tpe)
    expt.main(sys.argv)
