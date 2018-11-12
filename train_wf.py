import torch
import torch.nn as nn
import torch.optim as optim

import config as cnf

from torch.utils.data import DataLoader
from utils import unlabel_loss

from general_wf import GeneralWF
from labelData import LabelDataset
from unlabelData import UnlabelDataset
from dukeSeqLabelData import DukeSeqLabelDataset

Batch = 128
SeqLength = 24 # 32
UnlabelBatch = 1  
learning_rate = 0.0005  # learning rate
Trainstep = 20000  # number of train() calls
Lamb = 0.1  # ?
Thresh = 0.005  # unlabel_loss threshold

Snapshot = 5000  # do a snapshot every Snapshot steps (save period)
TestIter = 10  # do a testing every TestIter steps
ShowIter = 1  # print to screen

saveModelName = 'facing'
test_label_file = '/datadrive/person/DukeMTMC/test_heading_gt.txt'

class TrainWF(GeneralWF):

    def __init__(self, workingDir, prefix="", suffix="", device=None):
        super(TrainWF, self).__init__(workingDir, prefix, suffix, device)

        self.visualize = False
        self.labelBatch = Batch
        self.unlabelBatch = UnlabelBatch
        self.seqLength = SeqLength
        self.optimizer = optim.Adam(self.model.parameters(), lr=Lr)

        # counters
        self.labelEpoch = 0
        self.unlabelEpoch = 0
        self.countTrain = 0

        # Train dataset & loader
        label_dataset = LabelDataset(
            balance=True, mean=self.mean, std=self.std)
        self.train_loader = DataLoader(
            label_dataset, batch_size=self.labelBatch, shuffle=True, num_workers=6)

        unlabel_dataset = UnlabelDataset(
            self.seqLength, balance=True, mean=self.mean, std=self.std)
        self.train_unlabel_loader = DataLoader(
            unlabel_dataset, batch_size=1, shuffle=True, num_workers=4)

        # Train iterators
        self.train_data_iter = iter(self.train_loader)
        self.train_unlabel_iter = iter(self.train_unlabel_loader)

        # self.AV, self.AVP ?
        self.AV['loss'].avgWidth = 100  # there's a default plotter for 'loss'

        # second param is the number of average data
        self.add_accumulated_value('label_loss', 100)
        self.add_accumulated_value('unlabel_loss', 100)
        self.add_accumulated_value('test_loss', 10)
        self.add_accumulated_value('test_label', 10)
        self.add_accumulated_value('test_unlabel', 10)

        self.AVP.append(WorkFlow.VisdomLinePlotter(
            "total_loss", self.AV, ['loss', 'test_loss'], [True, True]))
        self.AVP.append(WorkFlow.VisdomLinePlotter(
            "label_loss", self.AV, ['label_loss', 'test_label'], [True, True]))
        self.AVP.append(WorkFlow.VisdomLinePlotter("unlabel_loss", self.AV, [
                        'unlabel_loss', 'test_unlabel'], [True, True]))

    def get_test_dataset(self):
        return DukeSeqLabelDataset(labelfile=test_label_file,
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
        self.save_model(self.model, saveModelName +
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
        sample, self.train_data_iter, self.labelEpoch = self.next_sample(
            self.train_data_iter, self.train_loader, self.labelEpoch)

        sample_unlabel, self.train_unlabel_data_iter, self.unlabelEpoch = self.next_sample(
            self.train_unlabel_data_iter, self.train_unlabel_loader, self.unlabelEpoch)

        # calculate loss
        label_loss = self.forward_label(sample)
        unlabel_loss = self.forward_unlabel(sample_unlabel)
        loss = label_loss + Lamb * unlabel_loss

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
            self.logger.info("%s #%d - (%d %d) %s lr: %.6f" % (exp_prefix[:-1],
                                                               self.countTrain, self.labelEpoch, self.unlabelEpoch, loss_str, learn))
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
