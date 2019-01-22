import sys
sys.path.append("..")

import os
import torch
import torch.optim as optim

from utils import get_path
from netWF import TrainWF
from dataset import *

from visdomPlotter import VisdomLinePlotter

Batch = 128
SeqLength = 24  # 32
UnlabelBatch = 1
LearningRate = 0.0005  # to tune
TrainStep = 200  # number of train() calls 20000

Snapshot = 500  # 500 model save period
TestIter = 50  # do a testing every TestIter steps
ShowIter = 50  # print to screen

ModelName = 'facing'

AccumulateValues = {"train_total": 100,
                    "train_label": 100,
                    "train_unlabel": 100,
                    "test_total": 20,
                    "test_label": 20,
                    "test_unlabel": 20}


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
                     "test_total": 20,
                     "test_label": 20,
                     "test_unlabel": 20}

        TrainWF.__init__(self, workingDir, prefix, ModelName,
                         trainStep=TrainStep, testIter=TestIter, saveIter=Snapshot, showIter=ShowIter, lr=LearningRate)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.add_plotter("total_loss", ['train_total', 'test_total'], [True, True])
        self.add_plotter("label_loss", ['train_label', 'test_label'], [True, True])
        self.add_plotter("unlabel_loss", ['train_unlabel', 'test_unlabel'], [True, True])

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

        self.test_dataset = DukeSeqLabelDataset("test-dukeseq", data_file=get_path(
            'DukeMTMC/train/val.csv'), seq_length=SeqLength, data_aug=True, mean=self.mean, std=self.std)
        self.test_loader = self.test_loader = DataLoader(self.test_dataset, batch_size=1)

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
        loss["total"].backward()
        self.optimizer.step()

        # update training loss history
        self.AV['train_total'].push_back(loss["total"].item(), self.countTrain)
        self.AV['train_label'].push_back(loss["label"].item(), self.countTrain)
        self.AV['train_unlabel'].push_back(loss["unlabel"].item(), self.countTrain)

    def test(self):
        """ update test loss history """
        self.logger.info("validation")

        # test one batch by calculating next sample loss
        WorkFlow.test(self)
        self.model.eval() # activate

        sample = self.test_loader.next_sample()
        inputs = sample[0].squeeze()
        targets = sample[1].squeeze()
        loss = self.model.forward_combine(inputs, targets, inputs)

        self.AV['test_total'].push_back(loss["total"].item(), self.countTrain)
        self.AV['test_label'].push_back(loss["label"].item(), self.countTrain)
        self.AV['test_unlabel'].push_back(loss["unlabel"].item(), self.countTrain)
