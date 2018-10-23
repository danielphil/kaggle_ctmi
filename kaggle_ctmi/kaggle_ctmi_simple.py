"""
A maximally simple solution to CT / CTA detection!
"""

import time

import keras
import numpy as np
from sklearn.model_selection import train_test_split

from algorithm import PreprocessedCohort, build_model, AccuracyHistory, save_model, data_scaling
from cohort import ShaipWorkspace, Cohort


def generate_static_index_html(shaip, filename):
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

<img src="example_images.png">

<p></p>

A summary table can be seen <a href="summary.html">  here </a>.

<h2>Results</h2>

See all results in the <a href="notebook.html"> Jupuyter Notebook</a>

</body>
</html>
"""
    with open(shaip.results_dir + filename, 'w') as fp:
        fp.write(contents)


# noinspection PyStringFormat
def main(shaip):
    np.random.seed(42)
    start = time.time()
    # Define our SHAIP Workspace file structure

    # Obtain the cohort of data from SHAIP
    cohort = Cohort(shaip)

    # Show and output images and information on what we're working with
    cohort.show_images(shaip.results_dir + 'example_images.png')
    cohort.explore_cohort(shaip.results_dir + 'summary.html')

    # Perform pre-processing (not yet cached)
    ppch = PreprocessedCohort(cohort)

    # Prepare for training - scaling and making a train/test split
    x_data = data_scaling(ppch.images)
    y_data = keras.utils.to_categorical(ppch.groundtruth, 2)
    ids = ppch.ids
    x_train, x_test, y_train, y_test, ids_train, ids_test = \
        train_test_split(x_data, y_data, ids,
                         stratify=y_data, test_size=0.20, shuffle=True, random_state=43)
    print("Training set: %d class 0, %d class 1" % (np.sum(y_train[:, 0]), np.sum(y_train[:, 1])))
    print("Testing set:  %d class 0, %d class 1" % (np.sum(y_test[:, 0]), np.sum(y_test[:, 1])))

    # Build the CNN model
    input_shape = PreprocessedCohort.imshape
    model = build_model(input_shape)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    history = AccuracyHistory()

    # Train and save the model
    model.fit(x_train, y_train,
              batch_size=20,
              shuffle=True,
              epochs=15,
              verbose=2,
              validation_split=0.2,
              callbacks=[history])
    save_model(model, shaip.models_dir + 'model')

    # Show some results of training
    history.plot_training()
    score = model.evaluate(x_test, y_test, verbose=0)

    # Output some results
    result = 'Test accuracy: %5.3f' % score[1]
    print(result)
    with open(shaip.results_dir + 'score.txt', 'w') as scorefp:
        scorefp.write(result)
    generate_static_index_html(shaip, 'index.html')
    print("Done in %4.1f seconds" % (time.time() - start))


# Lets do it!
if __name__ == '__main__':
    main(ShaipWorkspace('ShaipWorkspace/'))
