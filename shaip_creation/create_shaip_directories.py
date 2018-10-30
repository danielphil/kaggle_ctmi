"""
This script should be run once upon installation to setup the required input directories.
Input:  ./dicom_dir/*.dcm
Outputs: ./ShaipWorkspace/inputs/data
         ./ShaipWorkspace/inputs/groundtruth
Each of the data and groundtruth directories will have a folder per dataset, named by the
dataset id - e.g. ID_0001, containing the dicom file and GT respectively.

"""

import os
import shutil
from glob import glob

DICOM_DIR = 'dicom_dir'
SHAIP_INPUT_DIR = 'ShaipWorkspace/inputs'
GT_DIR = 'ShaipWorkspace/inputs/groundtruth'
DATA_DIR = 'ShaipWorkspace/inputs/data'


def _filename_to_contrast_gtstring(fname):
    """ Filenames look like this: "ID_0087_AGE_0044_CONTRAST_0_CT.dcm """
    assert fname[17:25] == 'CONTRAST'
    c = fname[26]
    assert c in ('0', '1')
    return 'ct\n' if c == '0' else 'cta\n'


def _filename_to_id(fname):
    """ Filenames look like this: "ID_0087_AGE_0044_CONTRASICOM_DIR = 'dicom_dir'
SHAIP_INPUT_DIR = 'ShaipWorkspace/inputs'
GT_DIR = 'ShaipWorkspace/inputs/groundtruth'
DATA_DIR = 'ShaipWorkspace/inputs/groundtruth'T_0_CT.dcm """
    assert fname[:3] == 'ID_'
    return fname[:7]


def main():
    assert os.path.isdir(DICOM_DIR)
    assert os.path.isdir(SHAIP_INPUT_DIR)

    assert not os.path.exists(GT_DIR)
    assert not os.path.exists(DATA_DIR)

    # Create the directories
    os.makedirs(GT_DIR)
    os.makedirs(DATA_DIR)

    # Glob the datasets
    datasets_paths = glob('dicom_dir/*.dcm')
    datasets_paths.sort()
    assert len(datasets_paths) == 100

    # Iterate of datasets, creating the folder in data and groundtruth
    # then copying in the data and a simple gt file - content 'ct' or 'cta'
    for dpath in datasets_paths:
        print("Processing", dpath)
        _, dname = os.path.split(dpath)
        id_ = _filename_to_id(dname)
        gt = _filename_to_contrast_gtstring(dname)
        gtpath = os.path.join(GT_DIR, id_)
        datapath = os.path.join(DATA_DIR, id_)

        assert not os.path.exists(gtpath)
        assert not os.path.exists(datapath)

        this_gt_dir = os.path.join(GT_DIR, id_)
        this_data_dir = os.path.join(DATA_DIR, id_)
        this_gt_path = os.path.join(this_gt_dir, id_ + '.txt')
        this_data_path = os.path.join(this_data_dir, id_ + '.dcm')

        print(this_gt_path)
        print(this_data_path)
        print()

        os.makedirs(this_gt_dir)
        os.makedirs(this_data_dir)

        shutil.copyfile(dpath, this_data_path)
        with open(this_gt_path, 'w') as f:
            f.write(gt)

        assert os.path.exists(this_gt_path)
        assert os.path.exists(this_data_path)

    print('Done')


if __name__ == '__main__':
    main()
