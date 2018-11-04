# kaggle_ctmi
Working with DICOM images from the [Kaggle CT Medical Images](https://www.kaggle.com/kmader/siim-medical-images).

This is one of the simplest DICOM based datasets I could find. I'm taking the goal to be prediction of CT vs CTA.
My goal is to create a simple clean, short solution with a minimum of added 'infrastructure'

## Recent improvements
* Images and GT read from directory structure as intended
* Logging to console and log file (using standard `logging` library)
* Basic dockerfile, image pushed to Docker Hub
* Lazy caching of pre-processed images as .npy files
* Command-line parameters to enable/disable experiment phases (using standard `argparse` library).

## Installation

Assuming a basic python 3.5 installation, roughly:
* clone into directory `kaggle_ctmi/`
* `cd kaggle.ctmi`
* `pip install -r requirments.txt`  (or use a virtual environment)
* `pytest` - all unit tests should pass out of the box.
* Setup the 'real'  'ShaipWorkspace':
  * `mkdir -p ShaipWorkspace/inputs`
  * `mkdir -p ShaipWorkspace/outputs/results`
  * `python shaip_creation/populate_shaip_directories.py`
  
* Run the main script:
  * `python kaggle_ctmi/experiment.py -tpe`
  
* To see results point a browser at `ShaipWorkspace/outputs/results/index.html`

## Usage
```
usage: experiment.py [-h] [-t] [-p] [-e]

CT/CTA discrimination to run in SHAIP

optional arguments:
  -h, --help      show this help message and exit
  -t, --train     perform model training
  -p, --predict   perform prediction over the test set
  -e, --evaluate  generate results

If no phases are specified, program does nothing - exits
```