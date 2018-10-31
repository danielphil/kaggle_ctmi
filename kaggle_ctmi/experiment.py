"""
A simple solution to CT / CTA detection based in Kaggle datasets.
"""

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
        self.algorithm = Algorithm()
        self.results = Results(self.shaip.results_dir)

    def main(self):
        """ Main Experiment entry point """
        print("Loading data...")
        cohort = Cohort(self.shaip)
        train_cohort, test_cohort = cohort.split_cohort_train_test(0.3)

        print("Training...")
        model = self.algorithm.train(train_cohort)

        self.algorithm.save_model(model, self.shaip.models_dir + 'model')

        print("Prediction...")
        test_predictions = self.algorithm.predict(model, test_cohort)

        print("Generating results to ShaipWorkspace/outputs/results/index.html...")
        self.results.show_results(train_cohort, test_cohort,
                                  self.algorithm.history, test_predictions)


# Lets do it!
if __name__ == '__main__':
    print("Starting Kaggle-CTMI Experiment\n")
    np.random.seed(42)
    start_time = time.time()

    expt = Experiment('ShaipWorkspace/')
    expt.main()

    print("Done in %4.1f seconds" % (time.time() - start_time))
