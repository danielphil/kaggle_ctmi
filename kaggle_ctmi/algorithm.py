"""
A maximally simple solution to CT / CTA detection!
"""

import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, model_from_json
from skimage.transform import downscale_local_mean


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
