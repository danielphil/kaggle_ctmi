
import numpy as np

from kaggle_ctmi.cohort import Cohort, ShaipWorkspace


def test_dicom_path_from_id():
    shaip = ShaipWorkspace()
    path = shaip.dicom_path_from_id('ID_0012')
    print(path)


def test_gt_path_from_id():
    shaip = ShaipWorkspace()
    path = shaip.gt_path_from_id('ID_0012')
    assert path == 'ShaipUnittestWorkspace/inputs/groundtruth/ID_0012/ID_0012.txt'


def test_read_contrast_gt():
    cohort = Cohort(ShaipWorkspace())
    gt_path = 'ShaipUnittestWorkspace/inputs/groundtruth/ID_0001/ID_0001.txt'
    gt = cohort._read_contrast_gt(gt_path)
    assert gt == 1


def test_init():
    cohort = Cohort(ShaipWorkspace())
    assert len(cohort.ids) == 16
    assert len(cohort.ids[0]) == 7 and cohort.ids[0][:3] == 'ID_'


def cohort_accessors_test_helper(cohort):
    assert len(cohort.dicoms) == len(cohort.ids) == len(cohort.images) == len(
        cohort.groundtruth) == len(cohort.groundtruth) == cohort.size
    assert all(['PixelData' in dcm for dcm in cohort.dicoms])
    assert len(cohort.images) == len(cohort.ids)
    assert all([im.shape == (512, 512) for im in cohort.images])
    assert all([im.dtype in (np.int16, np.uint16) for im in cohort.images])
    assert all([gt in (0, 1) for gt in cohort.groundtruth])
    c0, c1 = cohort.class_counts()
    assert c0 + c1 == cohort.size


def test_cohort_accessors():
    cohort = Cohort(ShaipWorkspace())
    cohort_accessors_test_helper(cohort)


def test_split_cohort_train_test():
    cohort = Cohort(ShaipWorkspace())
    test_prop = 0.25
    train_cohort, test_cohort = cohort.split_cohort_train_test(test_prop)
    n = cohort.size
    n_train = int(n * (1.0 - test_prop))
    n_test = int(n * test_prop)
    assert n_train + n_test == n
    assert train_cohort.size == n_train
    assert test_cohort.size == n_test

    cohort_accessors_test_helper(train_cohort)
    cohort_accessors_test_helper(test_cohort)
    cohort_accessors_test_helper(cohort)
