"""
A maximally simple solution to CT / CTA detection!
"""


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
