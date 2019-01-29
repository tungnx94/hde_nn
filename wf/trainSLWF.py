import sys
sys.path.insert(0, '..')

import torch
import utils
import numpy as np 
from trainWF import TrainWF

class TrainSLWF(TrainWF):

    def prepare_dataset(self, dloader):
        train_dts, val_dts = self.load_dataset()

        self.train_loader = dloader.loader(train_dts, self.batch)
        self.val_loader = dloader.loader(val_dts, self.batch_val)

    def train_loss(self):
        # get next samples
        inputs, targets = self.train_loader.next_sample()
        loss = self.model.loss_label(inputs, targets, mean=True)
        return loss

    # TODO: port to implementing class
    def evaluate(self, inputs, targets):
        # return numpy array [loss_label, loss_unlabel, loss_total, acc/angle_diff]
        outputs = self.model(inputs)
        loss = self.model.loss_label(inputs, targets)
        metric = utils.eval(outputs, targets)
        values = [torch.mean(loss).item(), metric]

        return np.array(values)

    # TODO: port to implementing class
    def val_metrics(self):
        # on train set
        sample = self.train_loader.next_sample()
        v1 = self.evaluate(sample[0], sample[1])

        # on val set
        sample = self.val_loader.next_sample()
        v2 = self.evaluate(sample[0], sample[1])
        
        return np.concatenate((v1, v2))
