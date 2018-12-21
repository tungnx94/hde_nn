import sys
sys.path.append("..")

import torch

from ssWF import SSWF
from netWF import TestWF
from utils import angle_metric, get_path, seq_show
from dataset import FolderLabelDataset, FolderUnlabelDataset, DukeSeqLabelDataset

from visdomPlotter import VisdomLinePlotter

LabelSeqLength = 24  # 32

TestLabelFile = 'DukeMCMT/test_heading_gt.txt'
TestLabelImgFolder = 'val_drone'
TestUnlabelImgFolder = 'drone_unlabel_seq'

TestStep = 100 # number of test() calls, 5000
ShowIter = 10
Snapshot = 50

Visualize = True

class TestSSWF(TestWF, SSWF):

    def __init__(self, workingDir, prefix, trained_model):
        self.visualize = Visualize

        SSWF.__init__(self)
        TestWF.__init__(self, workingDir, prefix,
                        trained_model, testStep=TestStep, saveIter=Snapshot, showIter=ShowIter)

    def visualize_output(self, inputs, outputs):
        seq_show(inputs.cpu().numpy(), dir_seq=outputs.detach().cpu().numpy(),
                 scale=0.8, mean=self.mean, std=self.std)

    def load_model(self):
        return SSWF.load_model(self)


class TestLabelSeqWF(TestSSWF):  # Type 1

    def __init__(self, workingDir, prefix, trained_model):
        self.acvs = {"total": 20,
                     "label": 20,
                     "unlabel": 20}

        TestSSWF.__init__(self, workingDir, prefix, trained_model)

        self.AVP.append(VisdomLinePlotter(
            "total_loss", self.AV, ['total'], [True]))
        self.AVP.append(VisdomLinePlotter(
            "label_loss", self.AV, ['label'], [True]))
        self.AVP.append(VisdomLinePlotter(
            "unlabel_loss", self.AV, ['unlabel'], [True]))

    def get_test_dataset(self):
        return DukeSeqLabelDataset("duke-test", get_path(TestLabelFile), seq_length=LabelSeqLength, data_aug=True,
                                   mean=self.mean, std=self.std)

    def calculate_loss(self, val_sample):
        """ combined loss """
        inputs = val_sample['imgseq'].squeeze()
        targets = val_sample['labelseq'].squeeze()

        loss = self.model.forward_combine(inputs, targets, inputs) 
        
        if self.visualize:  # display
            outputs = self.model(inputs)
            self.visualize_output(inputs, outputs)

            angle_error, cls_accuracy = angle_metric(
                outputs.detach().cpu().numpy(), targets.cpu().numpy())
            print 'loss: {}, angle diff %.4f, accuracy %.4f'.format(loss["total"].item(), angle_error, cls_accuracy)
        
        return loss

    def test(self):
        loss = SSWF.test(self)
        self.AV['total'].push_back(loss["total"].item())
        self.AV['label'].push_back(loss["label"].item())
        self.AV['unlabel'].push_back(loss["unlabel"].item())


class TestFolderWF(TestSSWF):  # Type 2

    def __init__(self, workingDir, prefix, trained_model):
        self.acvs = {"label": 20}

        TestSSWF.__init__(self, workingDir, prefix, trained_model)

        self.AVP.append(VisdomLinePlotter(
            "label_loss", self.AV, ['label'], [True]))

    def get_test_dataset(self):
        self.testBatch = 50
        return FolderLabelDataset("val-drone", get_path(TestLabelImgFolder), data_aug=False,
                                  mean=self.mean, std=self.std)

    def calculate_loss(self, val_sample):
        """ label loss only """
        inputs = val_sample['img']
        targets = val_sample['label']

        loss = self.model.forward_label(inputs, targets)
        
        if self.visualize:
            outputs = outputs = self.model(inputs)
            self.visualize_output(inputs, outputs)
            angle_error, cls_accuracy = angle_metric(
                outputs.detach().cpu().numpy(), targets.cpu().numpy())
            print 'label-loss %.4f, angle diff %.4f, accuracy %.4f' % (loss, angle_error, cls_accuracy)
        
        return loss

    def test(self):
        loss = SSWF.test(self)
        self.AV['label'].push_back(loss.item())


class TestUnlabelSeqWF(TestSSWF):  # Type 3

    def __init__(self, workingDir, prefix, trained_model):
        self.acvs = {"unlabel": 20}

        TestSSWF.__init__(self, workingDir, prefix, trained_model)

        self.AVP.append(VisdomLinePlotter(
            "unlabel_loss", self.AV, ['unlabel'], [True]))

    def get_test_dataset(self):
        return FolderUnlabelDataset("drone-unlabel", get_path(TestUnlabelImgFolder), data_aug=False,
                                    include_all=True, mean=self.mean, std=self.std)

    def calculate_loss(self, val_sample):
        """ unlabel loss only """
        inputs = val_sample.squeeze()
        loss = self.model.forward_unlabel(inputs) 
        

        if self.visualize:
            outputs = self.model(inputs)
            self.visualize_output(inputs, outputs)
            print 'loss-unlabel %.4f' % (loss)

        return loss

    def test(self):
        loss = SSWF.test(self)
        self.AV['unlabel'].push_back(loss.item())
