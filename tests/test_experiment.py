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


class TestCommandLine(unittest.TestCase):

    def test_command_line_empty(self):
        expt = Experiment('ShaipUnittestWorkspace/')
        with self.assertRaises(SystemExit):
            # With no arguments there's nothing to do, so expect to print usage then exit
            expt.command_line(['experiment.py'])

    def test_command_line_train_only(self):
        expt = Experiment('ShaipUnittestWorkspace/')
        expt.command_line(['experiment.py', '--train'])
        self.assertEqual(expt.args.train, True)
        self.assertEqual(expt.args.predict, False)
        self.assertEqual(expt.args.evaluate, False)

    def test_command_line_all_phases(self):
        expt = Experiment('ShaipUnittestWorkspace/')
        expt.command_line(['experiment.py', '-tpe'])
        self.assertEqual(expt.args.train, True)
        self.assertEqual(expt.args.predict, True)
        self.assertEqual(expt.args.evaluate, True)

    def test_command_line_h(self):
        expt = Experiment('ShaipUnittestWorkspace/')
        with self.assertRaises(SystemExit):
            expt.command_line(['experiment.py', '-h'])

    def test_command_line_help(self):
        expt = Experiment('ShaipUnittestWorkspace/')
        with self.assertRaises(SystemExit):
            expt.command_line(['experiment.py', '-help'])

    def test_command_line_error(self):
        expt = Experiment('ShaipUnittestWorkspace/')
        with self.assertRaises(SystemExit):
            expt.command_line(['experiment.py', '--sillyoption'])


class TestExperimentMain(unittest.TestCase):
    def test_all_phases(self):
        expt = Experiment('ShaipUnittestWorkspace/')
        expt.main(['experiment.py', '-tpe'])

    def test_notrain(self):
        expt = Experiment('ShaipUnittestWorkspace/')
        expt.main(['experiment.py', '-pe'])

    def test_predict_only(self):
        expt = Experiment('ShaipUnittestWorkspace/')
        expt.main(['experiment.py', '-p'])

    def test_nothing(self):
        expt = Experiment('ShaipUnittestWorkspace/')
        with self.assertRaises(SystemExit):
            expt.main(['experiment.py'])

    def test_h(self):
        expt = Experiment('ShaipUnittestWorkspace/')
        with self.assertRaises(SystemExit):
            expt.main(['experiment.py', '-h'])
