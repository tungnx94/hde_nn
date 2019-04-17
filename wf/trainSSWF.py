# WF class for semi-supervised learning

import sys
sys.path.insert(0, '..')

import utils
import numpy as np 
from .trainWF import TrainWF

class TrainSSWF(TrainSSWF):

    def prepare_dataset(self, dloader):
        label_dts, unlabel_dts, val_dts, val_unlabel_dts = dloader.load_dataset(self.config["dataset"])
        self.train_loader = dloader.loader(label_dts, self.batch)
        self.train_unlabel_loader = dloader.loader(unlabel_dts)

        self.val_loader = dloader.loader(val_dts, self.batch)
        self.val_unlabel_loader = dloader.loader(val_unlabel_dts)

    def train_error(self, sample):
        loss = self.model.loss_combine(sample[0], sample[1], sample[2], mean=True)
        return loss[2]

    def next_train_sample(self):
        sample = self.train_loader.next_sample()
        seq = self.train_unlabel_loader.next_sample().squeeze()  # remove 0-dim (=1)        
        return (sample[0], sample[1], seq)
    
    def next_val_sample(self):
        sample = self.val_loader.next_sample()
        seq = self.val_unlabel_loader.next_sample().squeeze()  # remove 0-dim (=1)        

        return (sample[0], sample[1], seq)

    def val_metrics(self, sample):
        # sample = (inputs, targets, seq)
        loss = self.model.loss_combine(sample[0], sample[1], sample[2], mean=True)
        values = [loss[0].item(), loss[1].item(), loss[2].item()]

        outputs = self.model(sample[0])
        angle_loss = utils.angle_err(outputs, sample[1])
        values.append(angle_loss)

        return np.array(values)

class TrainSSWF2(TrainWF):

    def prepare_dataset(self, dloader):
        label_dts, unlabel_dts, val_dts = dloader.load_dataset(self.config["dataset"])

        self.train_loader = dloader.loader(label_dts, self.batch)
        self.train_unlabel_loader = dloader.loader(unlabel_dts)
        self.val_loader = dloader.loader(val_dts)

    def next_val_sample(self):
        sample = self.val_loader.next_sample()
        inputs = sample[0].squeeze() # squeeze() makes sense here because we validate on DukeSeq data
        targets = sample[1].squeeze()
        return (inputs, targets, inputs)
