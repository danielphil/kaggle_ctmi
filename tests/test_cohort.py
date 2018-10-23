
import os
from tempfile import TemporaryDirectory

import numpy as np

from cohort import Cohort, ShaipWorkspace

SMILY = u'\U0001F603'


def test_cohort_init():
    cohort = Cohort(ShaipWorkspace())
    # print(datasets.ids)
    # print(datasets.id_to_path_map)
    assert len(cohort.ids) == 16
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
    cohort = Cohort(ShaipWorkspace())
    with TemporaryDirectory() as tmp_dir:
        savefilepath = os.path.join(tmp_dir, 'cohort_table.png')
        cohort.explore_cohort(savefilepath)
        assert os.path.exists(savefilepath)
    print(SMILY, "test_explore_cohort passed.")


def test_show_images():
    cohort = Cohort(ShaipWorkspace())
    with TemporaryDirectory() as tmp_dir:
        savefilepath = os.path.join(tmp_dir, 'image_gallery.png')
        cohort.show_images(savefilepath)
        assert os.path.exists(savefilepath)
    print(SMILY, "test_show_images passed.")


def test__filename_to_contrast_gt():
    fname = 'ID_0087_AGE_0044_CONTRAST_0_CT.dcm'
    gt = Cohort._filename_to_contrast_gt(fname)
    assert gt == 0
    print(SMILY, "test__filename_to_contrast_gt passed.")
