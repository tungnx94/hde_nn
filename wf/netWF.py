import sys
sys.path.append("..")

import os
import torch

from datetime import datetime
from workflow import WorkFlow


class TrainWF(WorkFlow):

    def __init__(self, workingDir, prefix, modelName,
                 trained_model=None, device=None, trainStep=100, testIter=20, saveIter=50, showIter=10, lr=0.005):

        # create folders
        t = datetime.now().strftime('%m-%d_%H:%M')
        workingDir = os.path.join(workingDir, prefix + "_" + t)

        self.traindir = os.path.join(workingDir, 'train')
        self.modeldir = os.path.join(workingDir, 'models')

        for folder in [workingDir, self.traindir, self.modeldir]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        self.modelName = modelName
        self.lr = lr

        self.trainStep = trainStep
        self.testIter = testIter

        WorkFlow.__init__(self, "train.log", trained_model, device, saveIter=saveIter, showIter=showIter)

    def get_log_dir(self):
        return self.traindir

    def save_model(self):
        """ Save :param: model to pickle file """
        model_path = os.path.join(
            self.modeldir, self.modelName + '_' + str(self.countTrain) + ".pkl")
        torch.save(self.model.state_dict(), model_path)

    def finalize(self):
        """ save model and values after training """
        WorkFlow.finalize(self)
        self.save_snapshot()

    def save_snapshot(self):
        """ write accumulated values and save temporal model """
        self.save_accumulated_values()
        self.save_model()

        self.logger.info("Saved snapshot")

    def run(self):
        """ train on all samples """
        self.logger.info("Started training")

        for iteration in range(1, self.trainStep + 1):
            self.train()

            # output losses
            if iteration % self.showIter == 0:
                self.logger.info("#%d %s" % (iteration, self.get_log_str()))

            # save temporary model
            if iteration % self.saveIter == 0:
                self.save_snapshot()

            if iteration % self.testIter == 0:
                self.test()

        self.logger.info("Finished training")


class TestWF(WorkFlow):

    def __init__(self, workingDir, prefix, trained_model, device=None, testStep=200, saveIter=50, showIter=10):
        t = datetime.now().strftime('%m-%d_%H:%M')
        self.modeldir = os.path.join(
            workingDir, 'models')  # should exist already
        self.testdir = os.path.join(workingDir, 'validation', prefix + "_" + t)

        if not os.path.isdir(self.testdir):
            os.makedirs(self.testdir)

        trained_model = os.path.join(self.modeldir, trained_model)
        self.testStep = testStep

        WorkFlow.__init__(self, "test.log", trained_model, device, saveIter=saveIter, showIter=showIter)

    def get_log_dir(self):
        return self.testdir

    def finalize(self):
        """ save model and values after training """
        WorkFlow.finalize(self)
        self.save_accumulated_values()

    def run(self):
        self.logger.info("Started testing")

        for iteration in range(1, self.testStep + 1):
            self.test()

            if iteration % self.showIter == 0:
                self.logger.info("#%d %s" % (iteration, self.get_log_str()))

            # save temporary model
            if iteration % self.saveIter == 0:
                self.save_accumulated_values()

        self.logger.info("Finished testing")
