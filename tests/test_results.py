
import os

from cohort import ShaipWorkspace
from results import generate_static_index_html


def test_generate_static_index_html():
    shaip = ShaipWorkspace()
    generate_static_index_html(shaip, "Fantastic score of .999", 'gash_index.html')
    assert os.path.exists(shaip.results_dir + 'gash_index.html')
