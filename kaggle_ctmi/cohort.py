"""
A maximally simple solution to CT / CTA detection!
"""

import os
from collections import Counter
from glob import glob

import pandas as pd
import pydicom
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

    def class_counts(self):
        """ Return a 2-tuple of counts for class 0 and class 1 in the cohort """
        counter = Counter(self.groundtruth)
        assert counter[0] + counter[1] == self.size
        return counter[0], counter[1]

    def split_cohort_train_test(self, test_size=0.3):
        """ Create two cohorts from this one, for train and test.
        Share image objects"""
        filepaths, y_data = self.filepaths, self.groundtruth,
        filepaths_train, filepaths_test = \
            train_test_split(filepaths,
                             stratify=y_data, test_size=test_size, shuffle=True, random_state=43)

        train_cohort = Cohort(filepaths_train)
        test_cohort = Cohort(filepaths_test)
        print("Training set: %d class 0, %d class 1" % train_cohort.class_counts())
        print("Testing set:  %d class 0, %d class 1" % test_cohort.class_counts())

        return train_cohort, test_cohort


# noinspection PyTypeChecker


