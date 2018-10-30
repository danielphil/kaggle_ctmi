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
        """ The default parameter case is used for unit tests """
        self.data_dir =        rootdir + 'inputs/data/'
        self.groundtruth_dir = rootdir + 'inputs/groundtruth/'
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

    def dicom_path_from_id(self, id_):
        return os.path.join(self.data_dir, id_, id_ + '.dcm')

    def gt_path_from_id(self, id_):
        return os.path.join(self.groundtruth_dir, id_, id_ + '.txt')


class Cohort(object):
    """ 
    Manages a SHAIP-like cohort of datasets, finding which are available, reading data and GT.
    Deals only with the raw input data - no normalization happens here.  
    Accessors generally present lazy evaluation semantics.
    """

    def __init__(self, shaip, only_these_ids=None):
        """ Create a cohort from the given shaip directory structure.  If 2nd parameter
        is given it is used to select only those given dataset ids.
        """
        self.shaip = shaip
        if only_these_ids is None:
            # Scan the shaip inputs folder to find ids
            dicompaths = glob(os.path.join(shaip.data_dir, '*'))
            self.ids = [os.path.basename(p) for p in dicompaths]
        else:
            self.ids = only_these_ids

        self.size = len(self.ids)

        # Private cache storage
        self._images = self._dicoms = self._groundtruth = None

    @property
    def dicoms(self):
        """ Lazily read and return a list of dicom objects in the same order as self.ids """
        if self._dicoms is None:
            self._dicoms = [pydicom.dcmread(self.shaip.dicom_path_from_id(id_))
                            for id_ in self.ids]
        return self._dicoms

    @property
    def images(self):
        """ Lazily extract and a list of images (2d numpy arrays) in the same order as self.ids """
        if self._images is None:
            self._images = [dcm.pixel_array for dcm in self.dicoms]
        return self._images

    @property
    def groundtruth(self):
        """ Return a list of ground-truth values as {0, 1} integers in the same order as self.ids"""
        if self._groundtruth is None:
            self._groundtruth = [Cohort._read_contrast_gt(self.shaip.gt_path_from_id(id_))
                                 for id_ in self.ids]
        return self._groundtruth

    def class_counts(self):
        """ Return a 2-tuple of counts for class 0 and class 1 in the cohort """
        counter = Counter(self.groundtruth)
        assert counter[0] + counter[1] == self.size
        return counter[0], counter[1]

    def split_cohort_train_test(self, test_size=0.3):
        """ Create two cohorts from this one, for train and test.
        Share image objects"""
        ids, y_data = self.ids, self.groundtruth,
        ids_train, ids_test = \
            train_test_split(ids,
                             stratify=y_data, test_size=test_size, shuffle=True, random_state=43)

        train_cohort = Cohort(self.shaip, ids_train)
        test_cohort = Cohort(self.shaip, ids_test)
        print("Training set: %d class 0, %d class 1" % train_cohort.class_counts())
        print("Testing set:  %d class 0, %d class 1" % test_cohort.class_counts())

        return train_cohort, test_cohort

    @staticmethod
    def _read_contrast_gt(gtpath):
        """ Read the file in the groundtruth folder and return its GT status"""
        with open(gtpath, 'r') as f:
            s = f.read()
            assert s in ('ct\n', 'cta\n')
            return 0 if s == 'ct\n' else 1
