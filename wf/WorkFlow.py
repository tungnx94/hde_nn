from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import signal
import logging

import numpy as np

from av import AccumulatedValue

class WFException(Exception):

    def __init__(self, message, name=None):
        self.message = message
        self.name = name

    def describe(self):
        if (self.name is not None):
            desc = self.name + ": " + self.message
        else:
            desc = self.message

        return desc


class SigIntException(WFException):

    def __init__(self, message, name=None):
        super(SigIntException, self).__init__(message, name)


class WorkFlow(object):

    # If Ctrl-C is sent to this instance, this will be set to be True.
    SIG_INT = False
    IS_FINALISING = False

    def __init__(self, workingDir, prefix="", suffix="", logFilename=None):
        # Add the current path to system path
        self.workingDir = workingDir  # The working directory.
        self.prefix = prefix
        self.suffix = suffix

        self.logdir = os.path.join(self.workingDir, 'logdata')
        self.imgdir = os.path.join(self.workingDir, 'resimg')
        self.modeldir = os.path.join(self.workingDir, 'models')

        if (not os.path.isdir(self.workingDir)):
            os.makedirs(self.workingDir)

        if (not os.path.isdir(self.logdir)):
            os.makedirs(self.logdir)

        if (not os.path.isdir(self.imgdir)):
            os.makedirs(self.imgdir)

        if (not os.path.isdir(self.modeldir)):
            os.makedirs(self.modeldir)

        self.isInitialized = False

        # Accumulated value dictionary.
        self.AV = {"loss": AccumulatedValue("loss")}

        # Accumulated value Plotter.
        # self.AVP should be an object of class AccumulatedValuePlotter.
        # The child class is responsible to populate this member.
        self.AVP = []

        self.verbose = False

        if (logFilename is not None):
            self.logFilename = logFilename
        else:
            self.logFilename = self.prefix + "wf" + self.suffix + ".log"

        # Logger.
        # logging.basicConfig(datefmt = '%m/%d/%Y %I:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(levelname)s: %(message)s')
        streamHandler.setFormatter(formatter)

        self.logger.addHandler(streamHandler)

        logFilePathPlusName = os.path.join(self.logdir, self.logFilename)
        fileHandler = logging.FileHandler(
            filename=logFilePathPlusName, mode="w")
        fileHandler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        fileHandler.setFormatter(formatter)

        self.logger.addHandler(fileHandler)

        self.logger.info("WorkFlow created.")

    def add_accumulated_value(self, name, avgWidth=2):
        # Check if there is alread an ojbect which has the same name.
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

        # Check whether the working directory exists.
        if (False == os.path.isdir(self.workingDir)):
            # Directory does not exist, create the directory.
            os.mkdir(self.workingDir)

        if (True == self.isInitialized):
            # This should be an error.
            desc = "The work flow is already initialized."
            exp = WFException(desc, "initialize")
            raise(exp)

        # Initialize AVP.
        if (len(self.AVP) > 0):
            self.AVP[0].initialize()
            self.logger.info("AVP initialized.")

        # add prefix to AVP
        for avp in self.AVP:
            avp.title = self.prefix + avp.title

        self.isInitialized = True

        self.debug_print("initialize() get called.")

    def train(self):
        # Check the system-wide signal.
        self.check_signal()

        if (False == self.isInitialized):
            # This should be an error.
            desc = "The work flow is not initialized yet."
            exp = WFException(desc, "tain")
            raise(exp)

        self.debug_print("train() get called.")

    def test(self):
        # Check the system-wide signal.
        self.check_signal()

        if (False == self.isInitialized):
            # This should be an error.
            desc = "The work flow is not initialized yet."
            exp = WFException(desc, "test")
            raise(exp)

        self.debug_print("test() get called.")

    def finalize(self):
        WorkFlow.IS_FINALISING = True

        if (False == self.isInitialized):
            # This should be an error.
            desc = "The work flow is not initialized yet."
            exp = WFException(desc, "finalize")
            raise(exp)

        # Write the accumulated values.
        self.write_accumulated_values()
        self.draw_accumulated_values()

        self.logger.info("Accumulated values are written to %s." %
                         (self.workingDir + "/AccumulatedValues"))

        self.isInitialized = False

        self.debug_print("finalize() get called.")

        WorkFlow.IS_FINALISING = False

    def plot_accumulated_values(self):
        if (0 == len(self.AVP)):
            return

        for avp in self.AVP:
            avp.update()

    def write_accumulated_values(self, outDir=None):
        if (outDir is None):
            outDir = self.logdir

        if (False == os.path.isdir(outDir)):
            os.makedirs(outDir)

        if (sys.version_info[0] < 3):
            for av in self.AV.itervalues():
                av.dump(outDir, self.prefix, self.suffix)
        else:
            for av in self.AV.values():
                av.dump(outDir, self.prefix, self.suffix)

    def draw_accumulated_values(self, outDir=None):
        if (outDir is None):
            outDir = self.imgdir

        if (False == os.path.isdir(outDir)):
            os.makedirs(outDir)

        for avp in self.AVP:
            avp.write_image(outDir, self.prefix, self.suffix)

    def is_initialized(self):
        return self.isInitialized

    def debug_print(self, msg):
        if (True == self.verbose):
            print(msg)

    def compose_file_name(self, fn, ext=""):
        return self.workingDir + "/" + self.prefix + fn + self.suffix + "." + ext

    def check_signal(self):
        if (True == WorkFlow.SIG_INT):
            raise SigIntException("SIGINT received.", "SigIntExp")

    def print_delimeter(self, title="", c="=", n=10, leading="\n", ending="\n"):
        d = [c for i in range(int(n))]

        if (0 == len(title)):
            s = "".join(d) + "".join(d)
        else:
            s = "".join(d) + " " + title + " " + "".join(d)

        print("%s%s%s" % (leading, s, ending))

    def get_log_str(self):
        logstr = ''
        for key in self.AV.keys():
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
