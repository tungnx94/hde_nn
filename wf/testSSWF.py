import sys
sys.path.append("..")

import torch

from ssWF import SSWF
from netWF import TestWF
from utils import get_path, seq_show
from dataset import SingleLabelDataset, FolderUnlabelDataset, DukeSeqLabelDataset

from visdomPlotter import VisdomLinePlotter

LabelSeqLength = 24  # 32

TestLabelImgFolder = 'drone_label/val'

TestStep = 100 # number of test() calls, 5000
ShowIter = 10
Snapshot = 50

Visualize = True

class TestSSWF(TestWF, SSWF):

    def __init__(self, workingDir, prefix, modelType, trained_model):
        self.visualize = Visualize

        SSWF.__init__(self, modelType)
        TestWF.__init__(self, workingDir, prefix,
                        trained_model, testStep=TestStep, saveIter=Snapshot, showIter=ShowIter)

    def visualize_output(self, inputs, outputs):
        seq_show(inputs.cpu().numpy(), dir_seq=outputs.detach().cpu().numpy(),
                 scale=0.8, mean=self.mean, std=self.std)

    def load_model(self):
        return SSWF.load_model(self)


class TestLabelSeqWF(TestSSWF):  # Type 1

    def __init__(self, workingDir, prefix, modelType, trained_model):
        self.acvs = {"total": 20,
                     "label": 20,
                     "unlabel": 20}

        TestSSWF.__init__(self, workingDir, prefix, modelType, trained_model)

        self.AVP.append(VisdomLinePlotter(
            "total_loss", self.AV, ['total'], [True]))
        self.AVP.append(VisdomLinePlotter(
            "label_loss", self.AV, ['label'], [True]))
        self.AVP.append(VisdomLinePlotter(
            "unlabel_loss", self.AV, ['unlabel'], [True]))

    def get_test_dataset(self):
        return DukeSeqLabelDataset("duke-test", data_file=get_path('DukeMTMC/val/person.csv'), seq_length=LabelSeqLength, data_aug=True,
                                   mean=self.mean, std=self.std)

    def calculate_loss(self, val_sample):
        """ combined loss """
        inputs = val_sample[0].squeeze()
        targets = val_sample[1].squeeze()

        loss = self.model.forward_combine(inputs, targets, inputs) 
        return loss

    def test(self):
        loss = SSWF.test(self)
        self.AV['total'].push_back(loss["total"].item())
        self.AV['label'].push_back(loss["label"].item())
        self.AV['unlabel'].push_back(loss["unlabel"].item())


class TestLabelWF(TestSSWF):  # Type 2

    def __init__(self, workingDir, prefix, modelType, trained_model):
        self.acvs = {"label": 20}

        TestSSWF.__init__(self, workingDir, prefix, modelType, trained_model)

        self.AVP.append(VisdomLinePlotter(
            "label_loss", self.AV, ['label'], [True]))

    def get_test_dataset(self):
        self.testBatch = 50
        return SingleLabelDataset("val-drone", data_file=get_path('DRONE_label'), data_aug=False,
                                  mean=self.mean, std=self.std)

    def calculate_loss(self, val_sample):
        """ label loss only """
        loss = self.model.forward_label(val_sample[0], val_sample[1])
        return loss

    def test(self):
        loss = SSWF.test(self)
        self.AV['label'].push_back(loss.item())


class TestUnlabelSeqWF(TestSSWF):  # Type 3

    def __init__(self, workingDir, prefix, modelType, trained_model):
        self.acvs = {"unlabel": 20}

        TestSSWF.__init__(self, workingDir, prefix, modelType, trained_model)

        self.AVP.append(VisdomLinePlotter(
            "unlabel_loss", self.AV, ['unlabel'], [True]))

    def get_test_dataset(self):
        return FolderUnlabelDataset("drone-unlabel", get_path('DRONE_seq'), data_aug=False,
                                    include_all=True, mean=self.mean, std=self.std)

    def calculate_loss(self, val_sample):
        """ unlabel loss only """
        inputs = val_sample.squeeze()
        loss = self.model.forward_unlabel(inputs) 
        return loss

    def test(self):
        loss = SSWF.test(self)
        self.AV['unlabel'].push_back(loss.item())
