
import os
from tempfile import TemporaryDirectory

from cohort import ShaipWorkspace, Cohort
from results import generate_static_index_html, explore_cohort, show_images


def test_generate_static_index_html():
    shaip = ShaipWorkspace()
    generate_static_index_html(shaip, "Fantastic score of .999", 'gash_index.html')
    assert os.path.exists(shaip.results_dir + 'gash_index.html')


def test_explore_cohort():
    cohort = Cohort.from_shaip_workspace(ShaipWorkspace())
    with TemporaryDirectory() as tmp_dir:
        savefilepath = os.path.join(tmp_dir, 'cohort_table.png')
        explore_cohort(cohort, savefilepath)
        assert os.path.exists(savefilepath)


def test_show_images():
    cohort = Cohort.from_shaip_workspace(ShaipWorkspace())
    with TemporaryDirectory() as tmp_dir:
        savefilepath = os.path.join(tmp_dir, 'image_gallery.png')
        show_images(cohort, savefilepath)
        assert os.path.exists(savefilepath)