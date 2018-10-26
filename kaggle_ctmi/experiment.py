"""
A maximally simple solution to CT / CTA detection based in Kaggle datasets.
"""

import time

import numpy as np
from sklearn.metrics import accuracy_score

from algorithm import Algorithm
from cohort import ShaipWorkspace, Cohort
from results import generate_static_index_html


class Experiment(object):
    """ This is the top-level class, orchestrating train/test split of the cohort,
    training and evaluation.  However he details are all elsewhere"""
    def __init__(self, shaip_root_dir):
        self.shaip = ShaipWorkspace(shaip_root_dir)
        self.shaip.check()
        self.algorithm = Algorithm()
        np.random.seed(42)

    def show(self, cohort):
        """ Should be moved into results """
        # Show and output images and information on what we're working with
        cohort.show_images(self.shaip.results_dir + 'test_images.png')
        cohort.explore_cohort(self.shaip.results_dir + 'test_summary.html')

    def evaluate(self, test_cohort, predictions):

        score = accuracy_score(test_cohort.groundtruth, predictions)
        # Output some results
        result = 'Test accuracy: %5.3f' % score
        print(result)
        with open(self.shaip.results_dir + 'score.txt', 'w') as scorefp:
            scorefp.write(result)
        generate_static_index_html(self.shaip, result, 'index.html')

    def main(self):
        """ Experiment entry point """
        cohort = Cohort.from_shaip_workspace(self.shaip)

        train_cohort, test_cohort = cohort.split_cohort_train_test(0.3)

        model = self.algorithm.train(train_cohort)

        self.algorithm.save_model(model, self.shaip.models_dir + 'model')

        self.show(test_cohort)

        test_predictions = self.algorithm.predict(model, test_cohort)

        self.evaluate(test_cohort, test_predictions)


# Lets do it!
if __name__ == '__main__':
    np.random.seed(42)
    start_time = time.time()

    expt = Experiment('ShaipWorkspace/')
    expt.main()

    print("Done in %4.1f seconds" % (time.time() - start_time))
