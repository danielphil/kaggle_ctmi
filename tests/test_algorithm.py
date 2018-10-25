
import os
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

import algorithm
from cohort import Cohort, ShaipWorkspace


def test__preprocess_one_dicom():
    cohort = Cohort.from_shaip_workspace(ShaipWorkspace())
    ppch = algorithm.PreprocessedCohort(cohort)
    dcm1 = cohort.dicoms[0]
    image = ppch._preprocess_one_dicom(dcm1)
    assert image.shape == algorithm.PreprocessedCohort.imshape
    plt.imshow(image)
    plt.colorbar()
    plt.show()


def test_preprocessed_cohort_accessors():
    ppch = algorithm.PreprocessedCohort(Cohort.from_shaip_workspace(ShaipWorkspace()))
    assert len(ppch.images) == len(ppch.ids) == len(ppch.groundtruth) == ppch.size


def test_data_scaling():
    xs, ys = 64, 128
    im = np.random.uniform(size=(xs, ys), high=2000, low=-300)
    n = 3
    images = [im] * n  # test set of just 3 images
    x_data = algorithm.data_scaling(images)
    expected_shape = (n, xs, ys, 1)
    assert x_data.shape == expected_shape
    assert x_data.dtype == np.float32


def test_train():
    cohort = Cohort.from_shaip_workspace(ShaipWorkspace())
    model = algorithm.train(cohort)
    assert model is not None


def test_build_model():
    model = algorithm.build_model((128, 128))
    model.summary()


def test_model_save_and_load():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(5, 1)))
    with TemporaryDirectory() as dir_name:
        temp_file_name = os.path.join(dir_name, 'test_model')
        algorithm.save_model(model, temp_file_name)
        _ = algorithm.load_model(temp_file_name)


def test_accuracyhistory():
    history = algorithm.AccuracyHistory()

    # Simulate some training!
    history.on_train_begin()
    for epoch, acc, val_acc in zip([1, 2, 3, 4], [.6, .7, .75, .75], [.6, .65, .68, .65]):
        log = {'acc': acc, 'val_acc': val_acc}
        history.on_epoch_end(epoch, log)

    history.plot_training()
