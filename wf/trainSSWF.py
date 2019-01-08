import sys
sys.path.append("..")
import os

import torch
import torch.optim as optim

from utils import get_path
from ssWF import SSWF
from netWF import TrainWF
from dataset import LabelDataset, UnlabelDataset, DukeSeqLabelDataset, DataLoader

from visdomPlotter import VisdomLinePlotter

Batch = 128
SeqLength = 24  # 32
UnlabelBatch = 1
LearningRate = 0.0005  # to tune
TrainStep = 200  # number of train() calls 20000

Snapshot = 100  # 500 model save period
TestIter = 50  # do a testing every TestIter steps
ShowIter = 20  # print to screen

ModelName = 'facing'
TestLabelFile = 'DukeMCMT/test_heading_gt.txt'

AccumulateValues = {"train_total": 100,
                    "train_label": 100,
                    "train_unlabel": 100,
                    "test_total": 20,
                    "test_label": 20,
                    "test_unlabel": 20}


class TrainSSWF(TrainWF, SSWF):

    def __init__(self, workingDir, prefix, modelType,
                 mobile_model=None, trained_model=None):

        self.labelBatch = Batch
        self.unlabelBatch = UnlabelBatch
        self.seqLength = SeqLength

        self.acvs = {"train_total": 100,
                    "train_label": 100,
                    "train_unlabel": 100,
                    "test_total": 20,
                    "test_label": 20,
                    "test_unlabel": 20}

        SSWF.__init__(self, modelType, mobile_model)
        TrainWF.__init__(self, workingDir, prefix, ModelName, trained_model=trained_model, 
                        trainStep=TrainStep, testIter=TestIter, saveIter=Snapshot, showIter=ShowIter, lr=LearningRate)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.AVP.append(VisdomLinePlotter(
            "total_loss", self.AV, ['train_total', 'test_total'], [True, True]))
        self.AVP.append(VisdomLinePlotter(
            "label_loss", self.AV, ['train_label', 'test_label'], [True, True]))
        self.AVP.append(VisdomLinePlotter(
            "unlabel_loss", self.AV, ['train_unlabel', 'test_unlabel'], [True, True]))

    def load_model(self):
        return SSWF.load_model(self)

    def load_dataset(self):
        # Train dataset & loader
        label_dataset = LabelDataset(
            balance=True, mean=self.mean, std=self.std)
        self.train_loader = DataLoader(
            label_dataset, batch_size=self.labelBatch, num_workers=6)

        unlabel_dataset = UnlabelDataset(
            self.seqLength, balance=True, mean=self.mean, std=self.std)
        self.train_unlabel_loader = DataLoader(
            unlabel_dataset, batch_size=self.unlabelBatch, num_workers=4)

    def get_test_dataset(self):
        return DukeSeqLabelDataset("duke-test", get_path(TestLabelFile),
                                   seq_length=SeqLength, data_aug=True, mean=self.mean, std=self.std)

    def train(self):
        """ train model (one batch) """
        super(TrainWF, self).train()
        self.model.train()

        self.countTrain += 1
        # get next samples
        sample = self.train_loader.next_sample()
        sample_unlabel = self.train_unlabel_loader.next_sample().squeeze()

        # calculate loss
        loss = self.model.forward_combine(sample['img'], sample['label'], sample_unlabel)

        # backpropagate
        self.optimizer.zero_grad()
        loss["total"].backward()
        self.optimizer.step()

        # update training loss history
        # convert to numeric value ?
        self.AV['train_total'].push_back(loss["total"].item(), self.countTrain)
        self.AV['train_label'].push_back(loss["label"].item(), self.countTrain)
        self.AV['train_unlabel'].push_back(loss["unlabel"].item(), self.countTrain)

    def calculate_loss(self, val_sample):
        """ combined loss """
        inputs = val_sample['imgseq'].squeeze() # squeeze() might not needed ?
        targets = val_sample['labelseq'].squeeze()

        loss = self.model.forward_combine(inputs, targets, inputs) 
        return loss

    def test(self):
        """ update test loss history """
        self.logger.info("validation")
        
        loss = SSWF.test(self)
        self.AV['test_total'].push_back(loss["total"].item(), self.countTrain)
        self.AV['test_label'].push_back(loss["label"].item(), self.countTrain)
        self.AV['test_unlabel'].push_back(loss["unlabel"].item(), self.countTrain)

    def run(self):
        TrainWF.run(self)
