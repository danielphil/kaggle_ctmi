import logging
from kaggle_ctmi.experiment import Experiment


def test_logging():
    print()
    expt = Experiment('ShaipUnittestWorkspace/')
    expt.setup_logging()

    logging.info("Starting Kaggle-CTMI Experiment")
    logging.info("This should go to both the console and logfile")
    logging.debug("This should go to the logfile only")
    logging.warning("You have been warned!")


def test_main():
    expt = Experiment('ShaipUnittestWorkspace/')
    expt.main()
