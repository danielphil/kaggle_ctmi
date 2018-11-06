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
import pydicom


def _filename_to_contrast_gtstring(fname):
    """ Filenames look like this: "ID_0087_AGE_0044_CONTRAST_0_CT.dcm """
    assert fname[17:25] == 'CONTRAST'
    c = fname[26]
    assert c in ('0', '1')
    return 'ct\n' if c == '0' else 'cta\n'


def _filename_to_id(fname):
    """ Filenames look like this: "ID_0087_AGE_0044_CONTRAST_0_CT.dcm """
    assert fname[:3] == 'ID_'
    return fname[:7]

class Identifier:
    def __init__(self):
        self.identifiers = {}
        self.counters = {}

    def get(self, *key):
        counter_key = key[:-1]
        if key not in self.identifiers:
            if counter_key not in self.counters:
                self.counters[counter_key] = 0
            self.identifiers[key] = self.counters[counter_key]
            self.counters[counter_key] += 1

        return str(self.identifiers[key])


study_ids = Identifier()
series_ids = Identifier()
instance_ids = Identifier()


def _get_dicom_identifier(fname):
    dcm = pydicom.dcmread(fname)
    study_id = study_ids.get(dcm.StudyInstanceUID)
    series_id = series_ids.get(dcm.StudyInstanceUID, dcm.SeriesInstanceUID)
    instance_id = instance_ids.get(dcm.StudyInstanceUID, dcm.SeriesInstanceUID, dcm.SOPInstanceUID)
    return (study_id, series_id, instance_id)


def do_it(shaip_root, only_these_ids=None):
    dicom_dir = 'dicom_dir'
    shaip_input_dir = os.path.join(shaip_root, 'inputs')
    gt_dir = os.path.join(shaip_root, 'inputs/groundtruth')
    data_dir = os.path.join(shaip_root, 'inputs/dicom')

    assert os.path.isdir(dicom_dir)

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
            print('Skipping...', id_)
            continue
        study_uid, series_uid, instance_uid = _get_dicom_identifier(dpath)
        gt = _filename_to_contrast_gtstring(dname)
        gtpath = os.path.join(gt_dir)
        datapath = os.path.join(data_dir, study_uid, series_uid)

        gt_path = os.path.join(gtpath, 'labels.txt')
        this_data_path = os.path.join(datapath, instance_uid + '.dcm')

        print(gt_path)
        print(this_data_path)
        print()

        os.makedirs(gtpath, exist_ok=True)
        os.makedirs(datapath, exist_ok=True)

        shutil.copyfile(dpath, this_data_path)
        with open(gt_path, 'a') as f:
            f.write("{}: {}".format(this_data_path, gt))

        assert os.path.exists(gt_path)
        assert os.path.exists(this_data_path)

    print('Done')


if __name__ == '__main__':
    # First for the main workspace, with all 100 images
    do_it('ShaipWorkspace')

    prepare_unittest_workspace = False

    if prepare_unittest_workspace:
        # Following only used by Ian to create the ShaipUnittestWorkspace, which is committed.
        # Then for unit tests we select 8 + 8 datasets, balancing ct and cta
        unit_test_ids_1 = ['ID_000' + str(i) for i in range(8)]
        unit_test_ids_0 = ['ID_005' + str(i) for i in range(8)]
        unit_test_ids = unit_test_ids_1 + unit_test_ids_0
        print(unit_test_ids)
        assert len(unit_test_ids) == 16

        do_it('ShaipUnittestWorkspace', unit_test_ids)
