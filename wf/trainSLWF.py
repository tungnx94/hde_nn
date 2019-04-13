# WF classes for supervised learning
import sys
sys.path.insert(0, '..')

import utils
import numpy as np 
from .trainWF import TrainWF

class TrainSLWF(TrainWF):

    def prepare_dataset(self, dloader):
        train_dts, val_dts = dloader.load_dataset(self.config["dataset"])

        self.train_loader = dloader.loader(train_dts, self.batch)
        self.val_loader = dloader.loader(val_dts, self.batch_val)

    def next_train_sample(self):
        return self.train_loader.next_sample()

    def next_val_sample(self):
        return self.val_loader.next_sample()

    def train_error(self, sample):
        loss = self.model.loss_label(sample[0], sample[1], mean=True)
        return loss

    def val_metrics(self, sample):
        # sample = (inputs, targets)
        outputs = self.model(sample[0])
        loss = self.model.loss_label(sample[0], sample[1], mean=True).item()

        angle_loss = utils.angle_err(outputs, sample[1])
        values = [loss, angle_loss]

        return np.array(values)
