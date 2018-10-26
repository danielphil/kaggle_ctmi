import pydicom
import pickle
import os
import numpy as np
from glob import glob
from skimage.transform import downscale_local_mean
import scipy.misc

images_path = 'input/images'

def filename_to_contrast_gt(fname):
    """ Filenames look like this: "ID_0087_AGE_0044_CONTRAST_0_CT.dcm """
    assert fname[17:25] == 'CONTRAST'
    c = fname[26]
    assert c in ('0', '1')
    return int(c)

def preprocess_one_dicom(dcm):
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
    downsample_factor = (4, 4)
    image = downscale_local_mean(image, downsample_factor)

    return image

def preprocess_dicom(image_path):
    file_id = os.path.basename(image_path)[:7]
    dicom_image = pydicom.dcmread(image_path)
    image = preprocess_one_dicom(dicom_image)

    # perform downsampling and clipping to data range
    output_path = '{}/{}.pkl'.format(images_path, file_id)
    with open(output_path, 'wb') as file:
        pickle.dump(image, file)
        print("Wrote %s" % output_path)

    # save ground truth


if __name__ == '__main__':
    filepaths = glob('dicom_dir/*.dcm')
    for path in filepaths:
        preprocess_dicom(path)