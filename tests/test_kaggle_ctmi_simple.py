# coding: utf-8

from kaggle_ctmi_simple import *

import numpy as np
import os
from tempfile import TemporaryDirectory

SMILY = u'\U0001F603'


def test_cohort_init():
    cohort = Cohort(ShaipWorkspace())
    # print(datasets.ids)
    # print(datasets.id_to_path_map)
    assert len(cohort.ids) == 100
    assert len(cohort.ids[0]) == 7 and cohort.ids[0][:3] == 'ID_'
    assert os.path.exists(cohort.filepaths[0])
    print(SMILY, "test_cohort_init passed.")


def test_cohort_accessors():
    cohort = Cohort(ShaipWorkspace())
    assert len(cohort.dicoms) == len(cohort.ids) == len(cohort.images) == len(
        cohort.groundtruth) == len(cohort.groundtruth) == len(cohort.filepaths) == cohort.size
    assert all(['PixelData' in dcm for dcm in cohort.dicoms])
    assert len(cohort.images) == len(cohort.ids)
    assert all([im.shape == (512, 512) for im in cohort.images])
    assert all([im.dtype in (np.int16, np.uint16) for im in cohort.images])
    assert all([gt in (0, 1) for gt in cohort.groundtruth])
    print(SMILY, "test_cohort_accessors passed.")


def test_explore_cohort():
    explore_cohort(Cohort(ShaipWorkspace()))
    with TemporaryDirectory() as tmp_dir:
        savefilepath = os.path.join(tmp_dir, 'cohort_table.png')
        explore_cohort(Cohort(ShaipWorkspace()), savefilepath)
        assert os.path.exists(savefilepath)

    print(SMILY, "test_explore_cohort passed.")


def test_show_images():
    with TemporaryDirectory() as tmp_dir:
        savefilepath = os.path.join(tmp_dir, 'image_gallery.png')
        show_images(Cohort(ShaipWorkspace()), savefilepath)
        assert os.path.exists(savefilepath)
    print(SMILY, "test_show_images passed.")


def test__filename_to_contrast_gt():
    fname = 'ID_0087_AGE_0044_CONTRAST_0_CT.dcm'
    gt = Cohort._filename_to_contrast_gt(fname)
    assert gt == 0
    print(SMILY, "test__filename_to_contrast_gt passed.")


def test__preprocess_one_dicom():
    cohort = Cohort(ShaipWorkspace())
    ppch = PreprocessedCohort(cohort)
    dcm1 = cohort.dicoms[0]
    image = ppch._preprocess_one_dicom(dcm1)
    assert image.shape == PreprocessedCohort.imshape
    plt.imshow(image)
    plt.colorbar()
    plt.show()
    print(SMILY, "test__preprocess_one_dicom passed.")


def test_preprocessed_cohort_accessors():
    ppch = PreprocessedCohort(Cohort(ShaipWorkspace()))
    assert len(ppch.images) == len(ppch.ids) == len(ppch.groundtruth) == ppch.size
    print(SMILY, "test_preprocessed_cohort_accessors passed.")


def test_data_scaling():
    xs, ys = 64, 128
    im = np.random.uniform(size=(xs, ys), high=2000, low=-300)
    n = 3
    images = [im] * n  # test set of just 3 images
    x_data = data_scaling(images)
    expected_shape = (n, xs, ys, 1)
    assert x_data.shape == expected_shape
    assert x_data.dtype == np.float32


def test_build_model():
    model = build_model((128, 128))
    model.summary()
    print(SMILY, "test_build_model passed.")


def test_model_save_and_load():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(5, 1)))
    with TemporaryDirectory() as dir_name:
        temp_file_name = os.path.join(dir_name, 'test_model')
        save_model(model, temp_file_name)
        _ = load_model(temp_file_name)
    print(SMILY, "test_model_save_and_load passed.")


def test_accuracyhistory():
    history = AccuracyHistory()

    # Simulate some training!
    history.on_train_begin()
    for epoch, acc, val_acc in zip([1, 2, 3, 4], [.6, .7, .75, .75], [.6, .65, .68, .65]):
        log = {'acc': acc, 'val_acc': val_acc}
        history.on_epoch_end(epoch, log)

    history.plot_training()
    print(SMILY, "test_AccuraryHistory passed.")


def test_generate_static_index_html():
    shaip = ShaipWorkspace()
    generate_static_index_html(shaip, 'gash_index.html')
    assert os.path.exists(shaip.results_dir + 'gash_index.html')
    print(SMILY, "test_generate_static_index_html passed")
