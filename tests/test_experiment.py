import logging
import unittest

from kaggle_ctmi.experiment import Experiment


def test_logging():
    print()
    expt = Experiment('ShaipUnittestWorkspace/')
    expt.setup_logging()

    logging.info("Starting Kaggle-CTMI Experiment")
    logging.info("This should go to both the console and logfile")
    logging.debug("This should go to the logfile only")
    logging.warning("You have been warned!")

def test_command_line_empty():
    expt = Experiment('ShaipUnittestWorkspace/')
    testargs = ['experiment.py']
    with unittest.mock.patch('sys.argv', testargs):
        expt.command_line()
        assert expt.args.notrain == False

def test_command_line_notrain():
    expt = Experiment('ShaipUnittestWorkspace/')
    testargs = ['experiment.py', '--notrain']
    with unittest.mock.patch('sys.argv', testargs):
        expt.command_line()
        assert expt.args.notrain == True

def test_command_line_nt():
    expt = Experiment('ShaipUnittestWorkspace/')
    testargs = ['experiment.py', '-nt']
    with unittest.mock.patch('sys.argv', testargs):
        expt.command_line()
        assert expt.args.notrain == True

def test_command_line_help():
    expt = Experiment('ShaipUnittestWorkspace/')
    testargs = ['experiment.py', '--help']
    with unittest.mock.patch('sys.argv', testargs):
        try:
            expt.command_line()
        except SystemExit:
            pass

def test_command_line_error():
    expt = Experiment('ShaipUnittestWorkspace/')
    testargs = ['experiment.py', 'silly']
    with unittest.mock.patch('sys.argv', testargs):
        try:
            expt.command_line()
        except SystemExit:
            pass

def test_main():
    expt = Experiment('ShaipUnittestWorkspace/')
    testargs = ['experiment.py']
    with unittest.mock.patch('sys.argv', testargs):
        expt.main()

def test_main_notrain():
    expt = Experiment('ShaipUnittestWorkspace/')
    testargs = ['experiment.py', '--notrain']
    with unittest.mock.patch('sys.argv', testargs):
        expt.main()
