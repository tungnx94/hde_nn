import sys
sys.path.append("..")
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_path
from generalWF import GeneralWF
from dataset import LabelDataset, UnlabelDataset, DukeSeqLabelDataset, DataLoader

from visdomPlotter import VisdomLinePlotter

Batch = 128
SeqLength = 24  # 32
UnlabelBatch = 1
LearningRate = 0.0005  # to tune
Trainstep = 20000  # number of train() calls
Thresh = 0.005  # unlabel_loss threshold

Snapshot = 20  # 500 do a snapshot every Snapshot steps (save period)
TestIter = 10  # do a testing every TestIter steps
ShowIter = 5  # print to screen

SaveModelName = 'facing'
TestLabelFile = 'DukeMCMT/test_heading_gt.txt'

AccumulateValues = {"train_total": 100,
                    "train_label": 100,
                    "train_unlabel": 100,
                    "test_total": 10,
                    "test_label": 10,
                    "test_unlabel": 10}


class TrainWF(GeneralWF):

    def __init__(self, workingDir, prefix,
                 device=None, mobile_model=None, trained_model=None):
        super(TrainWF, self).__init__(workingDir, prefix,
                                      device, mobile_model, trained_model)

        self.visualize = False
        self.labelBatch = Batch
        self.unlabelBatch = UnlabelBatch
        self.seqLength = SeqLength
        self.lr = LearningRate

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.countTrain = 0

        # Train dataset & loader
        label_dataset = LabelDataset(
            balance=True, mean=self.mean, std=self.std)
        self.train_loader = DataLoader(
            label_dataset, batch_size=self.labelBatch, num_workers=6)

        unlabel_dataset = UnlabelDataset(
            self.seqLength, balance=True, mean=self.mean, std=self.std)
        self.train_unlabel_loader = DataLoader(
            unlabel_dataset, batch_size=self.unlabelBatch, num_workers=4)

        self.AV = {}
        # second param is the number of average data
        for key, val in AccumulateValues.items():
            self.add_accumulated_value(key, val)

        self.AVP.append(VisdomLinePlotter(
            "total_loss", self.AV, ['train_total', 'test_total'], [True, True]))
        self.AVP.append(VisdomLinePlotter(
            "label_loss", self.AV, ['train_label', 'test_label'], [True, True]))
        self.AVP.append(VisdomLinePlotter(
            "unlabel_loss", self.AV, ['train_unlabel', 'test_unlabel'], [True, True]))

    def get_test_dataset(self):
        return DukeSeqLabelDataset("duke-test", get_path(TestLabelFile),
                                   seq_length=SeqLength, data_aug=True, mean=self.mean, std=self.std)

    def finalize(self):
        """ save model and values after training """
        super(TrainWF, self).finalize()
        self.save_snapshot()

        self.logger.info("Saved snapshot")

    def save_model(self, model, name):
        """ Save :param: model to pickle file """
        model_path = os.path.join(self.modeldir, name + ".pkl")
        torch.save(model.state_dict(), model_path)

    def save_snapshot(self):
        """ write accumulated values and save temporal model """
        self.write_accumulated_values()
        self.draw_accumulated_values()
        self.plot_accumulated_values()
        self.save_model(self.model, SaveModelName + '_' + str(self.countTrain))

    def unlabel_loss(self, output, threshold):
        """
        :param output: network unlabel output tensor
        :return: unlabel loss tensor
        """
        unlabel_batch = output.shape[0]
        loss_unlabel = torch.Tensor([0]).to(self.device).float()
        threshold = torch.tensor(threshold).to(self.device).float()

        for ind1 in range(unlabel_batch - 5):  # try to make every sample contribute
            # randomly pick two other samples
            ind2 = random.randint(ind1 + 2, unlabel_batch - 1)  # big distance
            ind3 = random.randint(ind1 + 1, ind2 - 1)  # small distance

            diff_big = torch.sum(
                (output[ind1] - output[ind2]) ** 2).float() / 2.0
            diff_small = torch.sum(
                (output[ind1] - output[ind3]) ** 2).float() / 2.0

            cost = torch.max(diff_small - diff_big - threshold,
                             torch.tensor(0).to(self.device).float())
            loss_unlabel += cost

        return loss_unlabel

    def forward_unlabel(self, sample):
        """
        :param sample: unlabeled data
        :return: unlabel loss
        """
        inputValue = sample.squeeze().to(self.device)
        output = self.model(inputValue)

        loss = self.unlabel_loss(output, Thresh)
        return loss.to(self.device).float()

    def forward_label(self, sample):
        """
        :param sample: labeled data
        :return: label loss
        """
        inputValue = sample['img'].to(self.device)
        targetValue = sample['label'].to(self.device)

        output = self.model(inputValue)

        loss = self.criterion(output, targetValue)
        return loss

    def train(self):
        """ train model (one batch) """
        super(TrainWF, self).train()
        self.model.train()

        self.countTrain += 1

        # get next samples
        sample = self.train_loader.next_sample()
        sample_unlabel = self.train_unlabel_loader.next_sample()

        # calculate loss
        label_loss = self.forward_label(sample)
        unlabel_loss = self.forward_unlabel(sample_unlabel)

        loss = label_loss + self.lamb * unlabel_loss

        # backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update training loss history
        # convert to numeric value ?
        self.AV['train_total'].push_back(loss.item(), self.countTrain)
        self.AV['train_label'].push_back(label_loss.item(), self.countTrain)
        self.AV['train_unlabel'].push_back(unlabel_loss.item(), self.countTrain)

        # record current params
        if self.countTrain % ShowIter == 0:
            self.logger.info("#%d %s" % (self.countTrain, self.get_log_str()))

        # save temporary model
        if (self.countTrain % Snapshot == 0):
            self.save_snapshot()

    def test(self):
        """ update test loss history """
        self.logger.info("validation")

        loss = GeneralWF.test(self)
        self.AV['test_total'].push_back(loss["total"], self.countTrain)
        self.AV['test_label'].push_back(loss["label"], self.countTrain)
        self.AV['test_unlabel'].push_back(loss["unlabel"], self.countTrain)

    def run(self):
        """ train on all samples """
        for iteration in range(1, Trainstep + 1):
            self.train()

            if iteration % TestIter == 0:
                self.test()

        self.logger.info("Finished training")
