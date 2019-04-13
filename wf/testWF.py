import os

from datetime import datetime
from .workflow import WorkFlow
from dataset import DatasetLoader
from utils import create_folder

class TestWF(WorkFlow):

    def __init__(self, config):
        t = datetime.now().strftime("%m-%d_%H-%M")

        workingDir = config["model"]["trained"]["path"]
        modelName = config["model"]["trained"]["weights"].split(".")[0]
        dtsName = config["dataset"]["test"]["name"]

        self.modeldir = os.path.join(workingDir, "models") 

        self.testdir = os.path.join(workingDir, "test", t + "_" +  + "_" + dtsName)
        create_folder(self.testdir)

        self.saveFreq = config["save_freq"] # av save frequency
        self.showFreq = config["show_freq"] # log frequency
        self.batch = config["batch"]

        self.logdir = self.testdir
        self.logfile = config["log"]

        WorkFlow.__init__(self, config)

    def finalize(self):
        """ save model and values after training """
        WorkFlow.finalize(self)
        self.save_accumulated_values()

    def prepare_dataset(self, dloader):
        test_dts = dloader.load_dataset(self.config["dataset"])
        self.test_loader = dloader.loader(test_dts, self.batch)

    def run(self):
        self.logger.info("Started testing")
        self.model.eval()

        self.iteration = 0
        for sample in self.test_loader:
            self.check_signal()
            self.iteration += 1

            # update acvs
            self.evaluate(sample)

            # log
            if self.iteration % self.showFreq == 0:
                self.logger.info("#%d %s" % (self.iteration, self.get_log_str()))

            # save temporary values
            if self.iteration % self.saveFreq == 0:
                self.save_accumulated_values()

        self.logger.info("Finished testing")

    def evaluate(self, sample):
        """ update error history """
        losses = self.val_metrics(sample)

        for idx, av in enumerate(self.config['losses']):
            self.push_to_av(av, losses[idx], self.iteration)