import sys
sys.path.insert(0, '..')

import os
import sys
import signal
import json
import logging

from datetime import datetime
from exception import WFException
from accValue import AccumulatedValue
from avPlotter import AccumulatedValuePlotter
from visdomPlotter import VisdomLinePlotter
from network import ModelLoader

class WorkFlow(object):

    # If Ctrl-C is sent to this instance, this will be set to be True.
    SIG_INT = False
    IS_FINALISING = False

    def __init__(self, config, verbose=False, livePlot=False):
        # True to enable debug_print
        self.config = config
        self.verbose = verbose
        self.livePlot = livePlot 
        self.isInitialized = False

        # Accumulated value dictionary & plotter.
        self.AV = AccumulatedValue(config['acvs'])
        self.AVP = []        
        for plot in config['plots']:
            self.add_plotter(plot['name'], plot['values'], plot['average'])            

        ### Init logging
        # Console log
        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logging.DEBUG)
        streamHandler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s'))

        # File log
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)

        # Save config params
        cnfPath = os.path.join(self.logdir, 'config.json')
        with open(cnfPath, 'w') as fp:
            json.dump(self.config, fp)

        # Init loggers
        logFilePath = os.path.join(self.logdir, self.logfile)

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

        self.load_model()

    def run(self):
        pass

    def load_dataset(self):
        pass

    def prepare_dataset(self, dloader):
        pass

    def load_model(self):
        loader = ModelLoader()
        self.model = loader.load(self.config['model'])
        self.countTrain = self.model.countTrain

    def proceed(self):
        self.initialize()
        self.run()
        self.finalize()

    def add_plotter(self, name, avList, plot_average):
        if self.livePlot:
            plotter = VisdomLinePlotter(name, self.AV, avList, plot_average)
        else:
            plotter = AccumulatedValuePlotter(name, self.AV, avList, plot_average)

        self.AVP.append(plotter)

    def push_to_av(self, name, value, stamp=None):
        # Check if the name exists.
        if not (name in self.AV.keys()):
            # This is an error.
            desc = "No object is registered as %s." % (name)
            exp = WFException(desc, "push_to_av")
            raise(exp)

        # Retrieve the AccumulatedValue object.
        self.AV.push_value(name, value, stamp)

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

        self.logger.info("FINISHED")

        self.endTime = datetime.now()
        self.logger.info("Total time: {}".format(
            self.endTime - self.startTime))

        self.isInitialized = False
        WorkFlow.IS_FINALISING = False

        self.debug_print("finalize() get called.")

    def write_accumulated_values(self, outDir):
        self.AV.save_csv(self.logdir)

    def plot_accumulated_values(self):
        for avp in self.AVP:
            avp.update()

    def draw_accumulated_values(self, outDir):
        for avp in self.AVP:
            avp.write_image(outDir)

    def save_accumulated_values(self):
        self.write_accumulated_values(self.logdir)
        self.draw_accumulated_values(self.logdir)
        self.plot_accumulated_values()

    def is_initialized(self):
        return self.isInitialized

    def debug_print(self, msg):
        if self.verbose:
            print(msg)

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
