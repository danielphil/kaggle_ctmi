
import os

from cohort import ShaipWorkspace
from experiment import generate_static_index_html, main


def test_generate_static_index_html():
    shaip = ShaipWorkspace()
    generate_static_index_html(shaip, 'gash_index.html')
    assert os.path.exists(shaip.results_dir + 'gash_index.html')


def test_main():
    main(ShaipWorkspace())
