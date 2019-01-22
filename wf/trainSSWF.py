import sys
sys.path.append("..")

import os
import torch
import torch.optim as optim

from utils import get_path
from netWF import TrainWF
from dataset import *

Batch = 128
SeqLength = 24  # 32
UnlabelBatch = 1
LearningRate = 0.001  # to tune
TrainStep = 2000  # number of train() calls 20000
ValStep = 50

Snapshot = 500  # 500 model save period
ValFreq = 200  # do a valing every valFreq steps
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

        self.labelBatch = Batch
        self.unlabelBatch = UnlabelBatch
        self.seqLength = SeqLength

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.acvs = {"train_total": 100,
                     "train_label": 100,
                     "train_unlabel": 100,
                     "val_total": 20,
                     "val_label": 20,
                     "val_unlabel": 20}

        TrainWF.__init__(self, workingDir, prefix, ModelName,
                         trainStep=TrainStep, valFreq=valFreq, saveFreq=Snapshot, showFreq=ShowFreq, lr=LearningRate)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.add_plotter("total_loss", ['train_total', 'val_total'], [True, True])
        self.add_plotter("label_loss", ['train_label', 'val_label'], [True, True])
        self.add_plotter("unlabel_loss", ['train_unlabel', 'val_unlabel'], [True, True])

    def load_dataset(self):
        # Labeled
        train_duke = SingleLabelDataset("train-duke", data_file=get_path(
            'DukeMTMC/train/train.csv'), data_aug=True, mean=self.mean, std=self.std)
        #train_duke.resize()

        train_virat = SingleLabelDataset("train-virat", data_file=get_path(
            'VIRAT/person/train.csv'), data_aug=True, mean=self.mean, std=self.std)
        #train_virat.resize()

        train_manual = SingleLabelDataset("train-handlabeled", data_file=get_path(
            'handlabel/person.csv'), data_aug=True, mean=self.mean, std=self.std)
        #train_manual.resize()

        label_dataset = MixDataset("Training-label")
        label_dataset.add(train_duke)
        label_dataset.add(train_virat)
        label_dataset.add(train_manual)

        self.train_loader = DataLoader(
            label_dataset, batch_size=self.labelBatch, num_workers=4)

        # Unlabeled
        unlabel_duke = FolderUnlabelDataset(
            "duke-unlabel", img_dir=get_path("DukeMTMC/train/images_unlabel"), mean=self.mean, std=self.std)
        #unlabel_duke.resize()

        unlabel_ucf = FolderUnlabelDataset(
            "ucf-unlabel", img_dir=get_path("UCF"), mean=self.mean, std=self.std)
        #unlabel_ucf.resize()

        unlabel_drone = FolderUnlabelDataset(
            "drone-unlabel", img_dir=get_path("DRONE_seq"), mean=self.mean, std=self.std)
        #unlabel_drone.resize()

        unlabel_dataset = MixDataset("Training-unlabel")
        unlabel_dataset.add(unlabel_duke)
        unlabel_dataset.add(unlabel_ucf)
        unlabel_dataset.add(unlabel_drone)

        self.train_unlabel_loader = DataLoader(
            unlabel_dataset, batch_size=self.unlabelBatch, num_workers=4)

        self.val_dataset = DukeSeqLabelDataset("val-dukeseq", data_file=get_path(
            'DukeMTMC/train/val.csv'), seq_length=SeqLength, data_aug=True, mean=self.mean, std=self.std)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1)

    def train(self):
        """ train model (one batch) """
        super(TrainWF, self).train()
        self.model.train()

        self.countTrain += 1
        # get next samples
        inputs, targets = self.train_loader.next_sample()
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
        self.model.eval() # activate

        losses = []
        for count in range(self.valStep):
            sample = self.val_loader.next_sample()
            inputs = sample[0].squeeze()
            targets = sample[1].squeeze()
            loss = self.model.forward_combine(inputs, targets, inputs)

            losses.append(loss.unsqueeze(0))

        losses = torch.tensor(tuple(losses), dim=0)
        loss_mean = torch.mean(losses, dim=0)

        self.AV['val_label'].push_back(loss_mean[0].item(), self.countTrain)
        self.AV['val_unlabel'].push_back(loss_mean[1].item(), self.countTrain)
        self.AV['val_total'].push_back(loss_mean[2].item(), self.countTrain)
