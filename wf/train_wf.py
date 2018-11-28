import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim

import config as cnf

from utils.data import unlabel_loss, get_path

from general_wf import GeneralWF, DataLoader
from data.labelData import LabelDataset
from data.unlabelData import UnlabelDataset
from data.dukeSeqLabelData import DukeSeqLabelDataset

Batch = 128
SeqLength = 24  # 32
UnlabelBatch = 1
LearningRate = 0.0005  # to tune
Trainstep = 20000  # number of train() calls
Thresh = 0.005  # unlabel_loss threshold

Snapshot = 5000  # do a snapshot every Snapshot steps (save period)
TestIter = 10  # do a testing every TestIter steps
ShowIter = 1  # print to screen

SaveModelName = 'facing'
TestLabelFile = 'DukeMCMT/test_heading_gt.txt'

AccumulateValues = {"label_loss": 100,
                    "unlabel_loss": 100,
                    "test_loss": 10,
                    "test_label": 10,
                    "test_unlabel": 10}


class TrainWF(GeneralWF):

    def __init__(self, workingDir, prefix="", suffix="", device=None):
        super(TrainWF, self).__init__(workingDir, prefix, suffix, device)

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

        # self.AV, self.AVP ?
        self.AV['loss'].avgWidth = 100  # there's a default plotter for 'loss'

        # second param is the number of average data
        for key, val in AccumulateValues:
            self.add_accumulated_value(key, val)

        self.AVP.append(WorkFlow.VisdomLinePlotter(
            "total_loss", self.AV, ['loss', 'test_loss'], [True, True]))
        self.AVP.append(WorkFlow.VisdomLinePlotter(
            "label_loss", self.AV, ['label_loss', 'test_label'], [True, True]))
        self.AVP.append(WorkFlow.VisdomLinePlotter("unlabel_loss", self.AV, [
                        'unlabel_loss', 'test_unlabel'], [True, True]))

    def get_test_dataset(self):
        return DukeSeqLabelDataset(get_path(TestLabelFile),
                                   batch=UnlabelBatch, data_aug=True, mean=self.mean, std=self.std)

    def finalize(self):
        """ save model and values after training """
        super(TrainWF, self).finalize()
        self.save_snapshot()

        print "Saved snapshot"

    def save_model(self, model, name):
        """ Save :param: model to pickle file """
        model_name = self.prefix + name + self.suffix + '.pkl'
        torch.save(model.state_dict(), self.modeldir + '/' + model_name)

    def save_snapshot(self):
        """ write accumulated values and save temporal model """
        self.write_accumulated_values()
        self.draw_accumulated_values()
        self.save_model(self.model, SaveModelName +
                        '_' + str(self.countTrain))

    def forward_unlabel(self, sample):
        """
        :param sample: unlabeled data
        :return: unlabel loss
        """
        inputValue = sample.squeeze().to(self.device)
        output = self.model(inputValue)

        loss = unlabel_loss(output.numpy(), Thresh)
        return torch.tensor([loss])

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
        # convert loss to numeric value ?
        self.AV['loss'].push_back(loss.item())
        self.AV['label_loss'].push_back(label_loss.item())
        self.AV['unlabel_loss'].push_back(unlabel_loss.item())

        # record current params
        if self.countTrain % ShowIter == 0:
            loss_str = self.get_log_str()
            self.logger.info("%s #%d - (%d %d) %s" % (exp_prefix[:-1],
                                                      self.countTrain, self.train_loader.epoch, self.train_unlabel_loader.epoch, loss_str))
        # save temporary model
        if (self.countTrain % Snapshot == 0):
            self.save_snapshot()

    def test(self):
        """ update test loss history """
        loss = GeneralWF.test(self)

        self.AV['test_loss'].push_back(loss["total"], self.countTrain)
        self.AV['test_label'].push_back(loss["label"], self.countTrain)
        self.AV['test_unlabel'].push_back(loss["unlabel"], self.countTrain)

    def run():
        """ train on all samples """
        for iteration in range(Trainstep):
            self.train()

            if iteration % TestIter == 0:
                self.test()

        print "Finished training"
