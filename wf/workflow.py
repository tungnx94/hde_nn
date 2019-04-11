import sys
sys.path.insert(0, "..")

import os
import sys
import signal
import json
import logging

from datetime import datetime
from utils import create_folder
from network import ModelLoader

from .exception import WFException
from .accValue import AccumulatedValue
from .avPlotter import AccumulatedValuePlotter
from .visdomPlotter import VisdomLinePlotter


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
        self.AV = AccumulatedValue(config["acvs"])
        self.AVP = []
        for plot in config["plots"]:
            self.add_plotter(plot["name"], plot["values"], plot["average"])

        ### Init logging
        # Console log
        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logging.DEBUG)
        streamHandler.setFormatter(
            logging.Formatter("%(levelname)s: %(message)s"))

        # File log
        create_folder(self.logdir)

        # Save config params
        cnfPath = os.path.join(self.logdir, "config.json")
        with open(cnfPath, "w") as fp:
            json.dump(self.config, fp, indent=4)

        # Init loggers
        logFilePath = os.path.join(self.logdir, self.logfile) # self.logfile defined by sub-WF

        fileHandler = logging.FileHandler(filename=logFilePath, mode="w")
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s"))

        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.logger.addHandler(streamHandler)
        self.logger.addHandler(fileHandler)

        self.logger.info("WorkFlow created.")

        self.load_model()

    def run(self):
        pass

    def prepare_dataset(self, dloader):
        pass

    def load_model(self):
        loader = ModelLoader()
        mconfig = self.config["model"]
        self.model = loader.load(mconfig)
        self.countTrain = self.model.countTrain

        if self.countTrain > 0 and mconfig["trained"]["continue"]: # loaded a pretrained model -> update AVs
            AV_file = os.join(mconfig["trained"]["path"], "models", mconfig["trained"]["weights"])
            self.AV.load_csv(AV_file) 

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

    def push_to_av(self, name, value, stamp):
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

        # Initialize AVP.
        if len(self.AVP) > 0:
            for avp in self.AVP:
                avp.initialize()

            self.logger.info("AVP initialized.")

        self.isInitialized = True
        self.startTime = datetime.now()

        self.logger.info("WF initialized.")

    def finalize(self):
        WorkFlow.IS_FINALISING = True

        self.logger.info("FINISHED")

        self.endTime = datetime.now()
        self.logger.info("Total time: {}".format(
            self.endTime - self.startTime))

        self.isInitialized = False
        WorkFlow.IS_FINALISING = False

    def save_accumulated_values(self):
        # save AVs to .csv
        self.AV.save_csv(self.logdir)

        # save plots
        for avp in self.AVP:
            avp.write_image(self.logdir)

        # plot live
        for avp in self.AVP:
            avp.update()

    def is_initialized(self):
        return self.isInitialized

    def debug_print(self, msg):
        if self.verbose:
            print(msg)

    def check_signal(self):
        if (True == WorkFlow.SIG_INT):
            raise WFException("SIGINT received.", "SigIntExp")

    def get_log_str(self):
        logstr = ""
        for key in sorted(self.AV.keys()):
            try:
                logstr += "%s: %.5f " % (key, self.AV.last_avg(key))
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
