import os
import sys
import time
import signal
import logging
import numpy as np
from datetime import datetime

from accValue import AccumulatedValue


class WFException(Exception):

    def __init__(self, message, name=None):
        self.message = message
        self.name = name

    def __str__(self):
        if (self.name is not None):
            desc = self.name + ": " + self.message
        else:
            desc = self.message

        return desc


class WorkFlow(object):

    # If Ctrl-C is sent to this instance, this will be set to be True.
    SIG_INT = False
    IS_FINALISING = False

    def __init__(self, workingDir, prefix, logFilename=None, verbose=False):
        # True to enable debug_print
        self.verbose = verbose

        # Add the current path to system path
        self.prefix = prefix

        t = datetime.now().strftime('%m-%d_%H:%M')
        self.workingDir = os.path.join(workingDir, prefix + "_" + t)

        # create log folders
        self.logdir = os.path.join(self.workingDir, 'logdata')
        self.modeldir = os.path.join(self.workingDir, 'models')

        for folder in [self.workingDir, self.logdir, self.modeldir]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        self.isInitialized = False

        # Accumulated value dictionary.
        self.AV = {"loss": AccumulatedValue("loss")}

        # Accumulated value Plotter.
        self.AVP = []

        # Log handlers

        # logging.basicConfig(datefmt = '%m/%d/%Y %I:%M:%S')
        # Console log
        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logging.DEBUG)
        streamHandler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s'))

        # File log
        if (logFilename is not None):
            self.logFilename = logFilename
        else:
            self.logFilename = "train.log"
        logFilePath = os.path.join(self.logdir, self.logFilename)

        fileHandler = logging.FileHandler(filename=logFilePath, mode="w")
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s'))

        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.logger.addHandler(streamHandler)
        self.logger.addHandler(fileHandler)

        self.logger.info("WorkFlow created.")

    def add_accumulated_value(self, name, avgWidth=2):
        # Check if there is alread an ojbect whifch has the same name.
        if (name in self.AV.keys()):
            # This is an error.
            desc = "There is already an object registered as \"%s\"." % (name)
            exp = WFException(desc, "add_accumulated_value")
            raise(exp)

        # Name is new. Create a new AccumulatedValue object.
        self.AV[name] = AccumulatedValue(name, avgWidth)

    def have_accumulated_value(self, name):
        return (name in self.AV.keys())

    def push_to_av(self, name, value, stamp=None):
        # Check if the name exists.
        if (False == (name in self.AV.keys())):
            # This is an error.
            desc = "No object is registered as %s." % (name)
            exp = WFException(desc, "push_to_av")
            raise(exp)

        # Retrieve the AccumulatedValue object.
        av = self.AV[name]

        av.push_back(value, stamp)

    def initialize(self):
        # Check the system-wide signal.
        self.check_signal()

        if self.isInitialized:
            # This should be an error.
            desc = "The work flow is already initialized."
            exp = WFException(desc, "initialize")
            raise(exp)

        # Initialize AVP.
        if len(self.AVP) > 0:
            self.AVP[0].initialize()
            self.logger.info("AVP initialized.")

        self.isInitialized = True
        self.startTime = datetime.now()

        self.logger.info("WF initialized.")
        self.debug_print("initialize() get called.")

    def train(self):
        # Check the system-wide signal.
        self.check_signal()

        if (False == self.isInitialized):  # only train after initilized
            # This should be an error.
            desc = "The work flow is not initialized yet."
            exp = WFException(desc, "tain")
            raise(exp)

        self.debug_print("train() get called.")

    def test(self):
        # Check the system-wide signal.
        self.check_signal()

        if (False == self.isInitialized):  # only test after initialized
            # This should be an error.
            desc = "The work flow is not initialized yet."
            exp = WFException(desc, "test")
            raise(exp)

        self.debug_print("test() get called.")

    def finalize(self):
        WorkFlow.IS_FINALISING = True

        if not self.isInitialized:
            # This should be an error.
            desc = "The work flow is not initialized yet."
            exp = WFException(desc, "finalize")
            raise(exp)

        # Write the accumulated values.
        self.save_accumulated_values()

        self.isInitialized = False
        WorkFlow.IS_FINALISING = False

        self.debug_print("finalize() get called.")

    def plot_accumulated_values(self):
        if len(self.AVP) == 0:
            return

        for avp in self.AVP:
            avp.update()

    def write_accumulated_values(self, outDir=None):
        if (outDir is None):
            outDir = self.logdir

        if (False == os.path.isdir(outDir)):
            os.makedirs(outDir)

        for av in self.AV.itervalues():
            av.save_csv(outDir)

    def draw_accumulated_values(self, outDir=None):
        if (outDir is None):
            outDir = self.workingDir

        if not os.path.isdir(outDir):
            os.makedirs(outDir)

        for avp in self.AVP:
            avp.write_image(outDir, self.prefix)

    def save_accumulated_values(self, outDir=None):
        self.write_accumulated_values(outDir)
        self.draw_accumulated_values(outDir)
        self.plot_accumulated_values()

    def is_initialized(self):
        return self.isInitialized

    def debug_print(self, msg):
        if self.verbose:
            print(msg)

    def compose_file_name(self, fn, ext=""):
        return self.workingDir + "/" + self.prefix + fn + "." + ext

    def check_signal(self):
        if (True == WorkFlow.SIG_INT):
            raise WFException("SIGINT received.", "SigIntExp")

    def print_delimeter(self, title="", c="=", n=10, leading="\n", ending="\n"):
        d = [c for i in range(int(n))]

        if (0 == len(title)):
            s = "".join(d) + "".join(d)
        else:
            s = "".join(d) + " " + title + " " + "".join(d)

        print("%s%s%s" % (leading, s, ending))

    def get_log_str(self):
        logstr = ""
        for key in sorted(self.AV.keys()):
            try:
                logstr += '%s: %.5f ' % (key, self.AV[key].last_avg())
            except WFException as e:
                continue

        return logstr

# Default signal handler.


def default_signal_handler(sig, frame):
    if (False == WorkFlow.IS_FINALISING):
        print("This is the default signal handler. Set SIG_INT flag for WorkFlow class.")
        WorkFlow.SIG_INT = True
    else:
        print("Receive SIGINT during finalizing! Abort the WorkFlow sequence!")
        sys.exit(0)

signal.signal(signal.SIGINT, default_signal_handler)
