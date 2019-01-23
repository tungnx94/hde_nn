import os
import torch

from datetime import datetime
from workflow import WorkFlow

class TrainWF(WorkFlow):

    def __init__(self, config):
        # create folders
        t = datetime.now().strftime('%m-%d_%H:%M')

        workingDir = os.path.join(config['dir'], config['prefix'] + "_" + t)
        self.traindir = os.path.join(workingDir, 'train')
        self.modeldir = os.path.join(workingDir, 'models')
        self.modelName = cnf['model']['name']
        self.logdir = self.traindir
        self.logfile = cnf['log']

        self.saveFreq = cnf['save_freq']
        self.showFreq = cnf['show_freq']
        self.valFreq = cnf['val_freq']
        self.trainStep = cnf['train_step']
        self.valStep = cnf['val_step']

        self.lr = cnf['lr']
        self.batch = cnf['batch']
        self.batch_unlabel = cnf['batch_unlabel']
        self.batch_val = cnf['batch_val']
        self.seq_length = cnf['seq_length']

        for folder in [workingDir, self.traindir, self.modeldir]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        WorkFlow.__init__(self, config)

    def save_model(self):
        """ Save :param: model to pickle file (pkl) """
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
        # WorkFlow.train(self)

        self.model.train()
        for iteration in range(1, self.trainStep + 1):
            WorkFlow.train(self)
            self.train()

            if iteration % self.valFreq == 0:
                WorkFlow.test(self)

                self.model.eval()
                self.validate()
                self.model.train()

            # output screen
            if iteration % self.showFreq == 0:
                self.logger.info("#%d %s" % (iteration, self.get_log_str()))

            # save temporary model
            if iteration % self.saveFreq == 0:
                self.save_snapshot()

        self.logger.info("Finished training")

    def train(self):
        pass

    def validate(self):
        pass



