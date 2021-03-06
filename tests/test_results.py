
import os
from tempfile import TemporaryDirectory

from kaggle_ctmi.cohort import Cohort, ShaipWorkspace
from kaggle_ctmi.results import Results


def test_generate_static_index_html():
    with TemporaryDirectory() as tmp_dir:
        results = Results(tmp_dir)
        score_text = "Fantastic score of .999"
        results.generate_static_index_html(score_text, 'index.html')
        assert os.path.exists(tmp_dir + 'index.html')


def test_explore_cohort():
    with TemporaryDirectory() as tmp_dir:
        results = Results(tmp_dir)
        cohort = Cohort(ShaipWorkspace())
        savefilename = 'cohort_table.png'
        results.explore_cohort(cohort, savefilename)
        assert os.path.exists(tmp_dir + savefilename)


def test_show_images():
    with TemporaryDirectory() as tmp_dir:
        results = Results(tmp_dir)
        cohort = Cohort(ShaipWorkspace())
        predictions = [0] * cohort.size
        savefilename = 'image_gallery.png'
        results.show_images(cohort, predictions, savefilename)
        assert os.path.exists(tmp_dir + savefilename)

