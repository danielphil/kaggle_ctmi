"""
A maximally simple solution to CT / CTA detection!
"""

import time

import keras
import numpy as np
from sklearn.model_selection import train_test_split

from algorithm import PreprocessedCohort, build_model, AccuracyHistory, save_model, data_scaling
from cohort import ShaipWorkspace, Cohort
from results import generate_static_index_html


class Experiment(object):
    def __init__(self, shaip_root_dir):
        self.shaip = ShaipWorkspace(shaip_root_dir)
        self.cohort = None
        self.x_data, self.y_data = None, None
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.ids_train, self.ids_test = None, None
        self.ids = None
        self.model = None

        np.random.seed(42)
        self.shaip.check()

    def load(self):
        # Obtain the cohort of data from SHAIP
        self.cohort = Cohort.from_shaip_workspace(self.shaip)

        # Show and output images and information on what we're working with
        self.cohort.show_images(self.shaip.results_dir + 'example_images.png')
        self.cohort.explore_cohort(self.shaip.results_dir + 'summary.html')

    def preprocess(self):
        # Perform pre-processing (not yet cached)
        ppch = PreprocessedCohort(self.cohort)
        # Prepare for training - scaling and making a train/test split
        self.x_data = data_scaling(ppch.images)
        self.y_data = keras.utils.to_categorical(ppch.groundtruth, 2)
        self.ids = self.cohort.ids

    # noinspection PyStringFormat
    def split_train_test(self):
        self.x_train, self.x_test, self.y_train, self.y_test, self.ids_train, self.ids_test = \
            train_test_split(self.x_data, self.y_data, self.ids,
                             stratify=self.y_data, test_size=0.20, shuffle=True, random_state=43)
        print("Training set: %d class 0, %d class 1" %
              (np.sum(self.y_train[:, 0]), np.sum(self.y_train[:, 1])))
        print("Testing set:  %d class 0, %d class 1" %
              (np.sum(self.y_test[:, 0]), np.sum(self.y_test[:, 1])))

    def train(self):
        # Build the CNN model
        input_shape = PreprocessedCohort.imshape
        self.model = build_model(input_shape)
        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])
        history = AccuracyHistory()

        # Train and save the model
        self.model.fit(
            self.x_train, self.y_train,
            batch_size=20, shuffle=True, epochs=15, verbose=2,
            validation_split=0.2, callbacks=[history])
        save_model(self.model, self.shaip.models_dir + 'model')
        history.plot_training()

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        # Output some results
        result = 'Test accuracy: %5.3f' % score[1]
        print(result)
        with open(self.shaip.results_dir + 'score.txt', 'w') as scorefp:
            scorefp.write(result)
        generate_static_index_html(self.shaip, 'index.html')

    def main(self):
        self.load()
        self.preprocess()
        self.split_train_test()
        self.train()
        self.evaluate()


# noinspection PyStringFormat
def old_main(shaip):
    np.random.seed(42)
    start = time.time()

    shaip.check()

    # Obtain the cohort of data from SHAIP
    cohort = Cohort(shaip)

    # Show and output images and information on what we're working with
    cohort.show_images(shaip.results_dir + 'example_images.png')
    cohort.explore_cohort(shaip.results_dir + 'summary.html')

    # Perform pre-processing (not yet cached)
    ppch = PreprocessedCohort(cohort)

    # Prepare for training - scaling and making a train/test split
    x_data = data_scaling(ppch.images)
    y_data = keras.utils.to_categorical(ppch.groundtruth, 2)
    ids = ppch.ids
    x_train, x_test, y_train, y_test, ids_train, ids_test = \
        train_test_split(x_data, y_data, ids,
                         stratify=y_data, test_size=0.20, shuffle=True, random_state=43)
    print("Training set: %d class 0, %d class 1" % (np.sum(y_train[:, 0]), np.sum(y_train[:, 1])))
    print("Testing set:  %d class 0, %d class 1" % (np.sum(y_test[:, 0]), np.sum(y_test[:, 1])))

    # Build the CNN model
    input_shape = PreprocessedCohort.imshape
    model = build_model(input_shape)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    history = AccuracyHistory()

    # Train and save the model
    model.fit(x_train, y_train,
              batch_size=20,
              shuffle=True,
              epochs=15,
              verbose=2,
              validation_split=0.2,
              callbacks=[history])
    save_model(model, shaip.models_dir + 'model')

    # Show some results of training
    history.plot_training()
    score = model.evaluate(x_test, y_test, verbose=0)

    # Output some results
    result = 'Test accuracy: %5.3f' % score[1]
    print(result)
    with open(shaip.results_dir + 'score.txt', 'w') as scorefp:
        scorefp.write(result)
    generate_static_index_html(shaip, 'index.html')
    print("Done in %4.1f seconds" % (time.time() - start))


# Lets do it!
if __name__ == '__main__':
    np.random.seed(42)
    expt = Experiment('ShaipWorkspace/')
    expt.main()
