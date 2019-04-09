# WF classes for supervised learning

import sys
sys.path.insert(0, '..')

import utils
import numpy as np 
from .trainWF import TrainWF

class TrainSLWF(TrainWF):

    def prepare_dataset(self, dloader):
        train_dts, val_dts = self.load_dataset()
        self.train_loader = dloader.loader(train_dts, self.batch)
        self.val_loader = dloader.loader(val_dts, self.batch_val)

    def next_sample(self, loader):
        return loader.next_sample()

    def next_train_sample(self):
        return self.next_sample(self.train_loader)

    def next_val_sample(self):
        return self.next_sample(self.val_loader)

    def train_error(self, sample):
        loss = self.model.loss_label(sample[0], sample[1], mean=True)
        return loss

    def val_metrics(self, sample):
        # sample = (inputs, targets)
        outputs = self.model(sample[0])
        loss = self.model.loss_label(sample[0], sample[1], mean=True).item()

        metric = utils.eval(outputs, sample[1])
        values = [loss, metric]

        return np.array(values)

class TrainRNNWF(TrainSLWF):

    def train_loss(self):
        # atm just cloned from superclass
        sample = self.get_next_sample(self.train_loader)
        loss = self.model.loss_label(sample[0], sample[1], mean=True)

        # uncomment this line will make a difference
        # loss = self.model.loss_weighted(sample[0], sample[1], mean=True)
        return loss

    def next_sample(self, loader):
        sample = loader.next_sample()
        return (sample[0].squeeze(), sample[1].squeeze())
