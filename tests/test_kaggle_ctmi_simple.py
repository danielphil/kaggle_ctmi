
import os

from cohort import ShaipWorkspace
from kaggle_ctmi_simple import generate_static_index_html, main

SMILY = u'\U0001F603'


def test_generate_static_index_html():
    shaip = ShaipWorkspace()
    generate_static_index_html(shaip, 'gash_index.html')
    assert os.path.exists(shaip.results_dir + 'gash_index.html')
    print(SMILY, "test_generate_static_index_html passed")


def test_main():
    main(ShaipWorkspace())
    print(SMILY, "test_main passed")
