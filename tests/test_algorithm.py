
from tempfile import TemporaryDirectory

from algorithm import *
from cohort import Cohort, ShaipWorkspace

SMILY = u'\U0001F603'


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