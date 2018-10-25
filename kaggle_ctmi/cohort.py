"""
A maximally simple solution to CT / CTA detection!
"""

import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from IPython.display import display

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
        self.cache_dir =       rootdir + 'cache/'                  # not yet used

    def check(self):
        for d in [self.data_dir, self.results_dir]:
            if not os.path.isdir(d):
                print("SHAIP directory %s is not found" % d)
                print("Working directory is %s", os.getcwd())
                assert False


class Cohort(object):
    """ 
    Manages a SHAIP-like cohort of datasets, finding which are available, reading data and GT.
    Deals only with the raw input data - no normalization happens here.  
    Accessors generally present lazy evaluation semantics.
    """

    def __init__(self, filepaths):
        """ This constructor takes a list of filepaths to DICOM files.  It can figure
        out groundtruth (contrast or not) from the filename"""
        self.filepaths = filepaths
        self.ids = [os.path.basename(fp)[:7] for fp in self.filepaths]
        self.size = len(self.ids)

        # Private cache storage
        self._images = self._dicoms = self._groundtruth = None

    @classmethod
    def from_shaip_workspace(cls, shaip):
        """ This constructor scans the data path to find what data is present and
        setup a list and dictionary of dataset ids and paths.  It does not *read*
        the data"""
        filepaths = glob(shaip.data_dir + '*.dcm')
        filepaths.sort()  # ensure order is deterministic
        return cls(filepaths)

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

    # def split_cohort_train_test(self, test_prop=0.3):
    #     """ Create two cohorts from this one, for train and test.
    #     Share image objects"""
    #     x_data, y_data, ids = self.images, self.groundtruth, self.ids
    #     x_train, x_test, y_train, y_test, ids_train, ids_test = \
    #         train_test_split(x_data, y_data, self.ids,
    #                          stratify=self.y_data, test_size=0.20, shuffle=True, random_state=43)
    #     print("Training set: %d class 0, %d class 1" %
    #           (np.sum(self.y_train[:, 0]), np.sum(self.y_train[:, 1])))
    #     print("Testing set:  %d class 0, %d class 1" %
    #           (np.sum(self.y_test[:, 0]), np.sum(self.y_test[:, 1])))

    # noinspection PyTypeChecker
    def explore_cohort(self, savefilepath=None):
        df = pd.DataFrame(
            columns=['ID', 'GT', 'Dtype', 'MinV', 'MaxV', 'Slope', 'Incpt', 'MmPerPix', 'Padding'])
        for ix in range(self.size):
            image = self.images[ix]
            dtype = image.dtype
            dcm = self.dicoms[ix]
            id_ = self.ids[ix]
            gt = self.groundtruth[ix]
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

    def show_images(self, savefilepath=None):
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
        for ix, ax in enumerate(axes.flat):  # Show just a selection
            im = self.images[ix]
            gt = self.groundtruth[ix]
            pltim = ax.imshow(im)
            ax.set_title("%s GT=%d" % (self.ids[ix], gt))
            fig.colorbar(pltim, ax=ax)
        if savefilepath is not None:
            _, extension = os.path.splitext(savefilepath)
            assert extension in ('.png', '.jpeg')
            plt.savefig(savefilepath)
        plt.show()
