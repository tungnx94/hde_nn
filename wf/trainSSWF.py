import sys
sys.path.append("..")

import os
import torch
import torch.optim as optim

from workflow import WorkFlow
from netWF import TrainWF

LearningRate = 0.001  # to tune
TrainStep = 20000  # number of train() calls 20000
ValStep = 50

Snapshot = 500  # 500 model save period
ValFreq = 200  # do a valing every valFreq steps 200
ShowFreq = 50  # print to screen

ModelName = 'facing'

AccumulateValues = {"train_total": 100,
                    "train_label": 100,
                    "train_unlabel": 100,
                    "val_total": 20,
                    "val_label": 20,
                    "val_unlabel": 20}


class TrainSSWF(TrainWF):

    def __init__(self, workingDir, prefix):

        self.acvs = AccumulateValues

        TrainWF.__init__(self, workingDir, prefix, ModelName,
                         trainStep=TrainStep, valFreq=ValFreq, saveFreq=Snapshot, showFreq=ShowFreq, lr=LearningRate)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.add_plotter("total_loss", ['train_total', 'val_total'], [True, True])
        self.add_plotter("label_loss", ['train_label', 'val_label'], [True, True])
        self.add_plotter("unlabel_loss", ['train_unlabel', 'val_unlabel'], [True, True])

    def train(self):
        """ train model (one batch) """
        self.countTrain += 1
        # get next samples
        inputs, targets, _ = self.train_loader.next_sample()
        unlabel_seqs = self.train_unlabel_loader.next_sample().squeeze()  # remove 0-dim (=1)

        # calculate loss
        loss = self.model.forward_combine(inputs, targets, unlabel_seqs)

        # backpropagate
        self.optimizer.zero_grad()
        loss[2].backward()
        self.optimizer.step()

        # update loss history
        self.AV['train_label'].push_back(loss[0].item(), self.countTrain)
        self.AV['train_unlabel'].push_back(loss[1].item(), self.countTrain)
        self.AV['train_total'].push_back(loss[2].item(), self.countTrain)

    def validate(self):
        """ update val loss history """
        self.logger.info("validation")

        # val one batch by calculating next sample loss
        WorkFlow.test(self)

        losses = []
        for count in range(self.valStep):
            sample = self.val_loader.next_sample()
            inputs = sample[0].squeeze()
            targets = sample[1].squeeze()
            loss = self.model.forward_combine(inputs, targets, inputs)

            losses.append(torch.tensor(loss).unsqueeze(0)) 

        losses = torch.cat(tuple(losses), dim=0)
        loss_mean = torch.mean(losses, dim=0)

        self.AV['val_label'].push_back(loss_mean[0].item(), self.countTrain)
        self.AV['val_unlabel'].push_back(loss_mean[1].item(), self.countTrain)
        self.AV['val_total'].push_back(loss_mean[2].item(), self.countTrain)
