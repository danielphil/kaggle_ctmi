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


def do_it(shaip_root, only_these_ids=None):
    dicom_dir = 'dicom_dir'
    shaip_input_dir = os.path.join(shaip_root, 'inputs')
    gt_dir = os.path.join(shaip_root, 'inputs/groundtruth')
    data_dir = os.path.join(shaip_root, 'inputs/data')

    assert os.path.isdir(dicom_dir)
    assert os.path.isdir(shaip_input_dir)

    assert not os.path.exists(gt_dir)
    assert not os.path.exists(data_dir)

    # Create the directories
    os.makedirs(gt_dir)
    os.makedirs(data_dir)

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
        if (only_these_ids is not None) and (id_ not in only_these_ids):
            print('Skippling...', id_)
            continue
        gt = _filename_to_contrast_gtstring(dname)
        gtpath = os.path.join(gt_dir, id_)
        datapath = os.path.join(data_dir, id_)

        assert not os.path.exists(gtpath)
        assert not os.path.exists(datapath)

        this_gt_dir = os.path.join(gt_dir, id_)
        this_data_dir = os.path.join(data_dir, id_)
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
    # First for the main workspace, with all 100 images
    do_it('ShaipWorkspace')

    if False:
        # Following only used by Ian to create the ShaipUnittestWorkspace, which is commited.
        # Then for unit tests we select 8 + 8 datasets, balancing ct and cta
        unit_test_ids_1 = ['ID_000' + str(i) for i in range(8)]
        unit_test_ids_0 = ['ID_005' + str(i) for i in range(8)]
        unit_test_ids = unit_test_ids_1 + unit_test_ids_0
        print(unit_test_ids)
        assert len(unit_test_ids) == 16

        do_it('ShaipUnittestWorkspace', unit_test_ids)
