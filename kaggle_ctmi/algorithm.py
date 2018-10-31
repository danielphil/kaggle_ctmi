"""
A maximally simple solution to CT / CTA detection!
"""

import logging
import os

import keras
import matplotlib
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, model_from_json
from skimage.transform import downscale_local_mean

matplotlib.use('Agg')
# noinspection PyPep8
import matplotlib.pyplot as plt


# noinspection PyMethodMayBeStatic,PyMethodMayBeStatic,PyMethodMayBeStatic
class Algorithm(object):
    """ This contains the details of our solution, intended to be largely
    isolated from other infra-structure issues. """

    def __init__(self):
        self.history = None     # Will keep a plot of accuracy by epoch

    # Class level constants
    downsample_factor = (4, 4)
    imshape = tuple(512 // dsf for dsf in downsample_factor)  # e.g. (128, 128)

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
        # logging.debug("Image (min,max) = (%6.1f, %6.1f)", np.min(image), np.max(image))
        clip_min = -200.0
        clip_max = 1000.0
        image[image < clip_min] = clip_min
        image[image > clip_max] = clip_max

        assert np.min(image) >= clip_min
        assert np.max(image) <= clip_max

        # Finally, downscale !

        image = downscale_local_mean(image, self.downsample_factor)

        return image

    def preprocessed_images(self, cohort):
        """ Apply preprocessing - mainly conversion to HU """
        logging.info("Preprocessing...")
        result = [self._preprocess_one_dicom(dcm) for dcm in cohort.dicoms]
        return result

    def train(self, cohort):
        """ Train on the given training cohort (already split from test)
        This includes pre-processing.   Return the trained model"""

        # Preprocess - two phases a) -> HU, b) reshape and scale.
        x_data = self.data_scaling(self.preprocessed_images(cohort))
        y_data = keras.utils.to_categorical(cohort.groundtruth, 2)

        # Build the model
        model = self.build_model()
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])
        self.history = AccuracyHistory()

        # Train and save the model
        model.fit(
            x_data, y_data,
            batch_size=20, shuffle=True, epochs=15, verbose=0,
            validation_split=0.2, callbacks=[self.history])

        return model

    def predict(self, model, cohort):
        # Preprocess
        x_data = self.data_scaling(self.preprocessed_images(cohort))

        # Run the model
        predictions = model.predict_classes(x_data)

        return predictions

    def data_scaling(self, images):
        """
        Given a list of pre-processed images (e.g. from PreprocessedCohort.images) perform
        intensity scaling and reshaping, returning a 4D tensor (n, x, y, 1) ready for feeding
        to a network
        """
        siz = images[0].shape
        x_data = np.array(images).reshape(-1, siz[0], siz[1], 1)
        x_data = x_data.astype(np.float32)
        x_data = (x_data + 100) / 150.0
        # mean, sd = np.mean(x_data), np.std(x_data)
        # min_, max_ = np.min(x_data), np.max(x_data)
        # print("data_scaling: shape:", x_data.shape, "min,max:",
        #      (min_, max_), "mean,sd:", (mean, sd))

        return x_data

    def build_model(self):
        input_shape = Algorithm.imshape + (1,)  # e.g. (128, 128, 1)
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

    def save_model(self, model, fname):
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
        logging.info("Model saved to %s[.json,.h5] files", fname)

    def load_model(self, fname):
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

    def plot_training(self, savefilepath):
        epochs = range(1, len(self.acc) + 1)
        plt.figure()
        plt.plot(epochs, self.acc, label='Train')
        plt.plot(epochs, self.val_acc, label='Validation')
        plt.ylim(0.0, 1.0)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        if savefilepath is not None:
            _, extension = os.path.splitext(savefilepath)
            assert extension in ('.png', '.jpeg')
            plt.savefig(savefilepath)
            plt.show()
        else:
            plt.show()
