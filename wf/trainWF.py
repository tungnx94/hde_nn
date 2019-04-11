import os
import torch
import numpy as np 
import torch.optim as optim 

from datetime import datetime
from .workflow import WorkFlow
from utils import create_folder

class TrainWF(WorkFlow):

    def __init__(self, config):
        # create folders
        t = datetime.now().strftime('%m-%d_%H:%M')

        # workingDir = os.path.join(config['dir'], config['prefix'] + "_" + t)
        workingDir = os.path.join(config['dir'], t + "_" + config["prefix"])

        self.traindir = os.path.join(workingDir, 'train')
        self.modeldir = os.path.join(workingDir, 'models')
        self.modelName = config['model']['name']
        self.logdir = self.traindir
        self.logfile = config['log']

        self.saveFreq = config['save_freq']
        self.showFreq = config['show_freq']
        self.valFreq = config['val_freq']
        self.trainStep = config['train_step']

        self.lr = config['lr']
        self.batch = config['batch']
        self.batch_unlabel = config['batch_unlabel']
        self.batch_val = config['batch_val']

        if 'lamb' in config:
            self.lamb = lamb

        for folder in [workingDir, self.traindir, self.modeldir]:
            create_folder(folder)

        WorkFlow.__init__(self, config)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def save_model(self):
        """ Save :param: model to pickle file (pkl) """
        model_path = os.path.join(
            self.modeldir, self.modelName + '_' + str(self.countTrain) + ".pkl")
        torch.save(self.model.state_dict(), model_path)

    def finalize(self):
        """ save model and values after training """
        WorkFlow.finalize(self)    
        self.save_snapshot()

        for avp in self.AVP:
            avp.write_image_final(self.logdir)

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
            self.check_signal()

            self.countTrain += 1
            train_sample = self.next_train_sample()

            # backward pass
            train_error = self.train_error(train_sample)
            self.optimizer.zero_grad()
            train_error.backward()
            self.optimizer.step()

            # Validation step
            if iteration % self.valFreq == 0:
                self.check_signal()
                self.model.eval()

                val_sample = self.next_val_sample()
                self.evaluate(train_sample, val_sample)

                self.model.train()

            # output screen
            if iteration % self.showFreq == 0:
                self.logger.info("#%d %s" % (iteration, self.get_log_str()))

            # save temporary model
            if iteration % self.saveFreq == 0:
                self.save_snapshot()

        self.logger.info("Finished training")

    def evaluate(self, train_sample, val_sample):
        """ update val loss history """
        train_losses = self.val_metrics(train_sample)
        val_losses = self.val_metrics(val_sample)

        losses = np.concatenate((train_losses, val_losses))

        for idx, av in enumerate(self.config['losses']):
            self.push_to_av(av, losses[idx], self.countTrain)

    def next_train_sample(self):
        pass

    def next_val_sample(self):
        pass

    def train_error(self):
        pass

    def val_metrics(self, sample):
        pass
