"""
A maximally simple solution to CT / CTA detection!
"""
import logging
import os

# Set the 'Agg' matplotlib backend to avoid plots appearing on the display (we only want them
# saved to .png files)
import matplotlib
import numpy as np
import pandas as pd
# from IPython.core.display import display
from sklearn.metrics import accuracy_score

matplotlib.use('Agg')

# noinspection PyPep8
from matplotlib import pyplot as plt


# noinspection PyMethodMayBeStatic,PyMethodMayBeStatic,PyMethodMayBeStatic
class Results(object):

    def __init__(self, results_dir):
        self.results_dir = results_dir

    def generate_static_index_html(self, result_text, filename):
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
    
    
    <h2>Training</h2>
    
    A table of training datasets can be seen <a href="training_cohort_table.html">  here </a>.
    
    <img src="training_images.png">
    
    <p></p>
    
    <img src="training_plot.png">
    <p></p>
    
    
    <h2>Test Results</h2>
    
    A table of test datasets can be seen <a href="test_cohort_table.html">  here </a>.
    
    <img src="test_images.png">
    <p></p>
    
    %s
    
    <p></p>
    
    ... more results to come!
    
    <h2> Log file </h2>
    The  log file can be seen <a href="kaggle-ctmi.log"> here </a>.
    
    </body>
    </html>
    """
        with open(self.results_dir + filename, 'w') as fp:
            fp.write(contents % result_text)

    def explore_cohort(self, cohort, savefilename=None):
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

        # display(df.describe(include='all'))
        # noinspection PyTypeChecker
        # display(df)
        if savefilename is not None:
            with open(self.results_dir + savefilename, 'w') as fp:
                df.to_html(fp)

    def show_images(self, cohort, predictions=None, savefilename=None):
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
        for ix, ax in enumerate(axes.flat):  # Show just a selection
            if ix >= len(cohort.images):
                break
            im = cohort.images[ix]
            gt = cohort.groundtruth[ix]
            pltim = ax.imshow(im)
            title = "%s GT=%d" % (cohort.ids[ix], gt)
            if predictions is not None:
                title += "; Pr=%d" % predictions[ix]
            ax.set_title(title)
            fig.colorbar(pltim, ax=ax)

        if savefilename is not None:
            _, extension = os.path.splitext(savefilename)
            assert extension in ('.png', '.jpeg')
            plt.savefig(self.results_dir + savefilename)
            plt.show()
        else:
            plt.show()

    def show_results(self, training_cohort, test_cohort, history, predictions):

        self.show_images(training_cohort, None, 'training_images.png')
        self.show_images(test_cohort, predictions, 'test_images.png')

        self.explore_cohort(training_cohort, 'training_cohort_table.html')
        self.explore_cohort(test_cohort, 'test_cohort_table.html')

        score = accuracy_score(test_cohort.groundtruth, predictions)

        # Output some results
        result = 'Test accuracy: %5.3f' % score
        logging.info(result)

        # Render the training plot to a png (if we did training on this run)
        if history:
            history.plot_training(os.path.join(self.results_dir, 'training_plot.png'))

        self.generate_static_index_html(result, 'index.html')
