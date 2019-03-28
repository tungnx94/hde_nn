# WF class for semi-supervised learning

import sys
sys.path.insert(0, '..')

import torch
import utils
import numpy as np 
from trainWF import TrainWF

class TrainSSWF(TrainWF):

    def prepare_dataset(self, dloader):
        label_dts, unlabel_dts, val_dts = self.load_dataset()
        self.train_loader = dloader.loader(label_dts, self.batch)
        self.train_unlabel_loader = dloader.loader(unlabel_dts, self.batch_unlabel)
        self.val_loader = dloader.loader(val_dts, self.batch_val)

    def train_loss(self):
        # get next samples
        inputs, targets = self.train_loader.next_sample()
        unlabel_seqs = self.train_unlabel_loader.next_sample().squeeze()  # remove 0-dim (=1)

        loss = self.model.loss_combine(inputs, targets, unlabel_seqs, mean=True)
        return loss[2] # total loss

    # TODO: port to implementing class
    def evaluate(self, inputs, targets, seq):
        # return numpy array [loss_label, loss_unlabel, loss_total, acc/angle_diff]
        outputs = self.model(inputs)
        loss = self.model.loss_combine(inputs, targets, seq)

        # values = [np.mean(loss[0].detach().numpy()), loss[1].item(), loss[2].item()]
        values = [torch.mean(loss[0]).item(), loss[1].item(), loss[2].item()]
        metric = utils.eval(outputs, targets)
        values.append(metric)

        return np.array(values)

    # TODO: port to implementing class
    def val_metrics(self):
        # on train set
        sample = self.train_loader.next_sample()
        seq = self.train_unlabel_loader.next_sample().squeeze()  # remove 0-dim (=1)        
        v1 = self.evaluate(sample[0], sample[1], seq)

        # on val set
        sample = self.val_loader.next_sample()
        inputs = sample[0].squeeze()
        targets = sample[1].squeeze()
        v2 = self.evaluate(inputs, targets, inputs)
        
        return np.concatenate((v1, v2))
