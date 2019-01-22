import os
import torch

from datetime import datetime
from workflow import WorkFlow


class TrainWF(WorkFlow):

    def __init__(self, workingDir, prefix, modelName,
                 trainStep=1000, valStep=100, valFreq=100, saveFreq=100, showFreq=25, lr=0.005):

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
        self.valStep = valStep
        self.valFreq = valFreq

        WorkFlow.__init__(self, "train.log", saveFreq=saveFreq, showFreq=showFreq)

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
            if iteration % self.showFreq == 0:
                self.logger.info("#%d %s" % (iteration, self.get_log_str()))

            # save temporary model
            if iteration % self.saveFreq == 0:
                self.save_snapshot()

            if iteration % self.valFreq == 0:
                self.validate()

        self.logger.info("Finished training")

    def validate(self):
        pass


class TestWF(WorkFlow):

    def __init__(self, workingDir, prefix, testStep=200, saveFreq=50, showFreq=25):
        t = datetime.now().strftime('%m-%d_%H:%M')
        self.modeldir = os.path.join(
            workingDir, 'models')  # should exist already
        self.testdir = os.path.join(workingDir, 'test', prefix + "_" + t)

        if not os.path.isdir(self.testdir):
            os.makedirs(self.testdir)

        self.testStep = testStep

        WorkFlow.__init__(self, "test.log", saveFreq=saveFreq, showFreq=showFreq)

    def get_log_dir(self):
        return self.testdir

    def finalize(self):
        """ save model and values after training """
        WorkFlow.finalize(self)
        self.save_accumulated_values()

    def run(self):
        self.logger.info("Started testing")

        WorkFlow.test(self)
        self.model.eval()

        for iteration in range(1, self.testStep + 1):
            self.test()

            if iteration % self.showFreq == 0:
                self.logger.info("#%d %s" % (iteration, self.get_log_str()))

            # save temporary model
            if iteration % self.saveFreq == 0:
                self.save_accumulated_values()

        self.logger.info("Finished testing")

    def test(self):
        pass
