import os
import time
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from kaggle_ctmi.algorithm import Algorithm, AccuracyHistory
from kaggle_ctmi.cohort import Cohort, ShaipWorkspace

# Set the 'Agg' matplotlib backend to avoid plots appearing on the display (we only want them
# saved to .png files)
matplotlib.use('Agg')
# noinspection PyPep8
import matplotlib.pyplot as plt


def test__preprocess_one_dicom():
    algorithm = Algorithm()
    cohort = Cohort(ShaipWorkspace())
    dcm1 = cohort.dicoms[0]
    image = algorithm._preprocess_one_dicom(dcm1)
    assert image.shape == Algorithm.imshape
    plt.imshow(image)
    plt.colorbar()
    plt.show()


def test_preprocessed_cohort():
    algorithm = Algorithm()
    cohort = Cohort(ShaipWorkspace())
    ppimages = algorithm.preprocessed_images(cohort)
    assert len(ppimages) == cohort.size

def test_preprocessed_cohort_with_cache():
    with TemporaryDirectory() as cache_dir:
        algorithm = Algorithm(cache_dir)
        cohort = Cohort(ShaipWorkspace())
        start1 = time.time()
        ppimages1 = algorithm.preprocessed_images(cohort)
        elapsed1 = time.time()-start1
        print("\nTime for first pass = %6.4f" % elapsed1)
        assert len(ppimages1) == cohort.size

        # And again, this time we should use the cache
        start2 = time.time()
        ppimages2 = algorithm.preprocessed_images(cohort)
        elapsed2 = time.time() - start2
        print("Time for second pass = %6.4f\n" % elapsed2)
        assert len(ppimages2) == cohort.size
        np.array_equal(ppimages1, ppimages2)

        # We expect the second pass to be much faster
        assert elapsed2 < elapsed1 / 5

def test_data_scaling():
    algorithm = Algorithm()
    xs, ys = 64, 128
    im = np.random.uniform(size=(xs, ys), high=2000, low=-300)
    n = 3
    images = [im] * n  # test set of just 3 images
    x_data = algorithm.data_scaling(images)
    expected_shape = (n, xs, ys, 1)
    assert x_data.shape == expected_shape
    assert x_data.dtype == np.float32


def test_train():
    algorithm = Algorithm()
    cohort = Cohort(ShaipWorkspace())
    model = algorithm.train(cohort)
    assert model is not None


def test_build_model():
    algorithm = Algorithm()
    model = algorithm.build_model()
    model.summary()


def test_model_save_and_load():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(5, 1)))
    with TemporaryDirectory() as dir_name:
        temp_file_path = os.path.join(dir_name, 'test_model')
        Algorithm.save_model(model, temp_file_path)
        _ = Algorithm.load_model(temp_file_path)


def test_accuracyhistory():
    history = AccuracyHistory()

    # Simulate some training!
    history.on_train_begin()
    for epoch, acc, val_acc in zip([1, 2, 3, 4], [.6, .7, .75, .75], [.6, .65, .68, .65]):
        log = {'acc': acc, 'val_acc': val_acc}
        history.on_epoch_end(epoch, log)

    with TemporaryDirectory() as dir_name:
        temp_file_path = os.path.join(dir_name, 'training_plot.png')
        history.plot_training(temp_file_path)
        assert os.path.exists(temp_file_path)
