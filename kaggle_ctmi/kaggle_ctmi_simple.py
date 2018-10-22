"""
A maximally simple solution to CT / CTA detection!
"""

import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import pydicom
import os
from glob import glob
import time

import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, model_from_json

from skimage.transform import downscale_local_mean
from sklearn.model_selection import train_test_split

pd.set_option('precision', 2)


# ## Loading DICOM from the SHAIP environment


class ShaipWorkspace(object):
    """ 
    This trivial class represents the shape (sic) of the SHAIP workspace,
    Defining where to find input datasets and GT, where to save results
    and models and where to find cache storage.  These are all Docker
    container local file paths.   
    """

    def __init__(self, rootdir='ShaipUnittestWorkspace/'):
        self.data_dir =        rootdir + 'inputs/dicomdata/'       # Will change
        self.groundtruth_dir = rootdir + 'inputs/groundtruth/'     # Not yet used
        self.results_dir =     rootdir + 'outputs/results/'
        self.models_dir =      rootdir + 'outputs/models/'
        self.tensorboad_dir =  rootdir + 'outputs/tensorboard/'    # Not yet used
        self.cache_dir =       rootdir + 'Scache/'                  # not yet used


class Cohort(object):
    """ 
    Manages a SHAIP-like cohort of datasets, finding which are available, reading data and GT.
    Deals only with the raw input data - no normalization happens here.  
    Accessors generally present lazy evaluation semantics.
    """

    def __init__(self, shaip):
        """ The constructor scans the data path to find what data is present and
        setup a list and dictionary of dataset ids and paths.  It does not *read*
        the data"""
        self.shaip = shaip
        self.filepaths = glob(self.shaip.data_dir + '*.dcm')
        self.ids = [os.path.basename(fp)[:7] for fp in self.filepaths]
        self.id_to_path_map = {id_: path for id_, path in zip(self.ids, self.filepaths)}
        self.size = len(self.ids)

        # Private cache storage
        self._images = self._dicoms = self._groundtruth = None

    @property
    def dicoms(self):
        """ Lazily read and return a list of dicom objects in the same order as self.ids """
        if self._dicoms is None:
            self._dicoms = [pydicom.dcmread(fp) for fp in self.filepaths]
        return self._dicoms

    @property
    def images(self):
        """ Lazily extract and a list of images (2d numpy arrays) in the same order as self.ids """
        if self._images is None:
            self._images = [dcm.pixel_array for dcm in self.dicoms]
        return self._images

    @staticmethod
    def _filename_to_contrast_gt(fname):
        """ Filenames look like this: "ID_0087_AGE_0044_CONTRAST_0_CT.dcm """
        assert fname[17:25] == 'CONTRAST'
        c = fname[26]
        assert c in ('0', '1')
        return int(c)

    @property
    def groundtruth(self):
        """ Return a list of ground-truth values as {0, 1} integers in the same order as self.ids"""
        if self._groundtruth is None:
            self._groundtruth = [Cohort._filename_to_contrast_gt(os.path.basename(fp)) for fp in
                                 self.filepaths]
        return self._groundtruth


# noinspection PyTypeChecker
def explore_cohort(cohort, savefilepath=None):
    df = pd.DataFrame(
        columns=['ID', 'GT', 'Dtype', 'MinV', 'MaxV', 'Slope', 'Incpt', 'MmPerPix', 'Padding'])
    for ix in range(cohort.size):
        image = cohort.images[ix]
        dtype = image.dtype
        dcm = cohort.dicoms[ix]
        id_ = cohort.ids[ix]
        gt = cohort.groundtruth[ix]
        padding = dcm.data_element(
            'PixelPaddingValue').value if 'PixelPaddingValue' in dcm else None
        slope = dcm.data_element('RescaleSlope').value
        intercept = dcm.data_element('RescaleIntercept').value
        min_, max_ = float(np.min(image)), float(np.max(image))
        mmpp_x, mmpp_y = dcm.data_element('PixelSpacing').value
        assert mmpp_x == mmpp_y
        row = (id_, gt, dtype, min_, max_, slope, intercept, mmpp_x, padding)

        df.loc[ix] = row

    display(df.describe(include='all'))
    display(df)
    if savefilepath is not None:
        with open(savefilepath, 'w') as fp:
            df.to_html(fp)


def show_images(cohort, savefilepath=None):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    for ix, ax in enumerate(axes.flat):  # Show just a selection
        im = cohort.images[ix]
        gt = cohort.groundtruth[ix]
        pltim = ax.imshow(im)
        ax.set_title("%s GT=%d" % (cohort.ids[ix], gt))
        fig.colorbar(pltim, ax=ax)
    if savefilepath is not None:
        _, extension = os.path.splitext(savefilepath)
        assert extension in ('.png', '.jpeg')
        plt.savefig(savefilepath)
    plt.show()


class PreprocessedCohort(object):
    """ 
    Represents cohort of data with basic pre-procession applied.  For example, deal with padding,
    conversion to Hounsfield etc.  At this stage we are no longer concerned with file formats,
    directories etc.
    """
    downsample_factor = (4, 4)
    imshape = tuple(512 // dsf for dsf in downsample_factor)  # e.g. (128, 128)

    def __init__(self, cohort):

        self.size = cohort.size  # Number of images
        self.ids = cohort.ids
        self.groundtruth = cohort.groundtruth
        self.dicoms = cohort.dicoms

        self._preprocessed_images = None

    def _preprocess_one_dicom(self, dcm):
        """ Return a nicely normalised numpy float32 image """
        raw_image = dcm.pixel_array

        # print(raw_image.dtype)
        slope = dcm.data_element('RescaleSlope').value
        intercept = dcm.data_element('RescaleIntercept').value

        image = np.array(raw_image, dtype=np.float32)
        image = image * slope + intercept
        image = np.array(image, dtype=np.float32)

        # It seems that padding value lies!  So we'll just clamp image values and hope for the best!
        # print("Image (min,max) = (%6.1f, %6.1f)" % (np.min(image), np.max(image)))
        clip_min = -200.0
        clip_max = 1000.0
        image[image < clip_min] = clip_min
        image[image > clip_max] = clip_max

        assert np.min(image) >= clip_min
        assert np.max(image) <= clip_max

        # Finally, downscale !

        image = downscale_local_mean(image, self.downsample_factor)

        return image

    @property
    def images(self):
        """ Lazily apply normalisation """
        if self._preprocessed_images is None:
            self._preprocessed_images = [self._preprocess_one_dicom(dcm) for dcm in self.dicoms]
        return self._preprocessed_images


def data_scaling(images):
    """
    Given a list of pre-processed images (e.g. from PreprocessedCohort.images) perform
    intensity scaling and reshaping, returning a 4D tensor (n, x, y, 1) ready for feeding
    to a network
    """

    siz = images[0].shape
    x_data = np.array(images).reshape(-1, siz[0], siz[1], 1)
    x_data = x_data.astype(np.float32)
    x_data = (x_data + 100) / 150.0
    mean, sd = np.mean(x_data), np.std(x_data)
    min_, max_ = np.min(x_data), np.max(x_data)
    print("data_scaling: shape:", x_data.shape, "min,max:", (min_, max_), "mean,sd:", (mean, sd))

    return x_data


def build_model(image_shape):
    input_shape = image_shape + (1,)  # e.g. (128, 128, 1)
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(8, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model


def save_model(model, fname):
    """ Save model and wieghts to fname and fname.h5 files respectively 
    fname can include a directory which will be created if it doesn't exist"""

    directory = os.path.dirname(fname)
    if directory and not os.path.isdir(directory):
        print("Creating directory %s" % directory)
        os.makedirs(directory)

    model_json = model.to_json()
    with open(fname + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(fname + '.h5')
    print("Model saved to %s[.json,.h5] files" % fname)


def load_model(fname):
    """ Load a model from fname.json and fname.h5, and return it. 
    (Note that the loaded model must be compiled before use)"""
    # load json and create model
    json_file = open(fname + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(fname + '.h5')
    print("Loaded model from %s[.json,.h5] files" % fname)
    return loaded_model


class AccuracyHistory(keras.callbacks.Callback):
    """ Record and plot training progress """

    def __init__(self):
        super().__init__()
        self.acc = []
        self.val_acc = []

    def on_train_begin(self, logs=None):
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

    def plot_training(self):
        epochs = range(1, len(self.acc) + 1)
        plt.plot(epochs, self.acc, label='Train')
        plt.plot(epochs, self.val_acc, label='Validation')
        plt.ylim(0.0, 1.0)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


def generate_static_index_html(shaip, filename):
    """ Given filename should include the path and filename. Write the file
    into the SHAIP results_dir directory """
    assert '.html' in filename
    # noinspection PyPep8
    contents = """<!DOCTYPE html>
<html>
<body>

<h1>Kaggle CTMI Example results</h1>
These are results for the problem of CT/CTA detection.

<h2>Data source</h2>

Datasets come from the 
<a href="https://www.kaggle.com/kmader/siim-medical-images/home">
Kaggle Medical Imaging dataset</a>.  

The data is originally from  
<a href="https://wiki.cancerimagingarchive.net/display/Public/TCGA-LUAD">
The Cancer Genome Atlas LUADA</a> collection

<h2> Dataset exploration</h2>

<img src="example_images.png">

<p></p>

A summary table can be seen <a href="summary.html">  here </a>.

<h2>Results</h2>

See all results in the <a href="notebook.html"> Jupuyter Notebook</a>

</body>
</html>
"""
    with open(shaip.results_dir + filename, 'w') as fp:
        fp.write(contents)


# noinspection PyStringFormat
def main(shaip):
    start = time.time()
    # Define our SHAIP Workspace file structure

    # Obtain the cohort of data from SHAIP
    cohort = Cohort(shaip)

    # Show and output images and information on what we're working with
    show_images(cohort, shaip.results_dir + 'example_images.png')
    explore_cohort(cohort, shaip.results_dir + 'summary.html')

    # Perform pre-processing (not yet cached)
    ppch = PreprocessedCohort(cohort)

    # Prepare for training - scaling and making a train/test split
    x_data = data_scaling(ppch.images)
    y_data = keras.utils.to_categorical(ppch.groundtruth, 2)
    ids = ppch.ids
    x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(x_data, y_data, ids,
                                                                             test_size=0.20,
                                                                             shuffle=True,
                                                                             random_state=21)
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
    main(ShaipWorkspace('ShaipWorkspace/'))
