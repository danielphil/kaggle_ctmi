"""
A maximally simple solution to CT / CTA detection!
"""
import os

import numpy as np
import pandas as pd
from IPython.core.display import display
from matplotlib import pyplot as plt


def generate_static_index_html(shaip, result_text, filename):
    """ Given filename should include the path and filename. Write the file
    into the SHAIP results_dir directory """
    assert '.html' in filename
    # noinspection PyPep8
    contents = """<!DOCTYPE html>
<html>
<body>

<h1>Kaggle CTMI Example results</h1>
These are results for the problem of CT/CTA detection.

<h2>Data source</h2>

Datasets come from the 
<a href="https://www.kaggle.com/kmader/siim-medical-images/home">
Kaggle Medical Imaging dataset</a>.  

The data is originally from  
<a href="https://wiki.cancerimagingarchive.net/display/Public/TCGA-LUAD">
The Cancer Genome Atlas LUADA</a> collection

<h2> Dataset exploration</h2>

<img src="test_images.png">

<p></p>

A summary table can be seen <a href="test_summary.html">  here </a>.

<h2>Results</h2>

%s

<p></p>

... more results to come!

</body>
</html>
"""
    with open(shaip.results_dir + filename, 'w') as fp:
        fp.write(contents % result_text)


def explore_cohort(cohort, savefilepath=None):
    df = pd.DataFrame(
        columns=['ID', 'GT', 'Dtype', 'MinV', 'MaxV', 'Slope', 'Incpt', 'MmPerPix', 'Padding'])
    for ix in range(cohort.size):
        image = cohort.images[ix]
        dtype = image.dtype
        dcm = cohort.dicoms[ix]
        id_ = cohort.ids[ix]
        gt = cohort.groundtruth[ix]
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


def show_images(cohort, savefilepath=None):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    for ix, ax in enumerate(axes.flat):  # Show just a selection
        if ix >= len(cohort.images):
            break
        im = cohort.images[ix]
        gt = cohort.groundtruth[ix]
        pltim = ax.imshow(im)
        ax.set_title("%s GT=%d" % (cohort.ids[ix], gt))
        fig.colorbar(pltim, ax=ax)

    if savefilepath is not None:
        _, extension = os.path.splitext(savefilepath)
        assert extension in ('.png', '.jpeg')
        plt.savefig(savefilepath)
    plt.show()