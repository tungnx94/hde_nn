import sys
sys.path.append("..")

import torch

from ssWF import SSWF
from netWF import TestWF

from utils import unlabel_loss_np, angle_metric, get_path
from dataset import FolderLabelDataset, FolderUnlabelDataset, DukeSeqLabelDataset

LabelSeqLength = 24  # 32
TestStep = 5000  # number of test() calls

TestLabelFile = 'DukeMCMT/test_heading_gt.txt'
TestLabelImgFolder = 'val_drone'
TestUnlabelImgFolder = 'drone_unlabel_seq'


class TestSSWF(TestWF, SSWF):

    def __init__(self, workingDir, prefix, trained_model, device=None):
        self.visualize = True

        SSWF.__init__(self)
        TestWF.__init__(self, workingDir, prefix, trained_model, device, testStep=TestStep)


class TestLabelSeqWF(TestSSWF):  # Type 1

    def __init__(self, workingDir, prefix, trained_model, device=None):
        self.acvs = {"total": 20,
                    "label": 20,
                    "unlabel": 20}

        TestSSWF.__init__(self, workingDir, prefix, trained_model, device)

        self.AVP.append(VisdomLinePlotter("total_loss", self.AV, ['total'], [True]))
        self.AVP.append(VisdomLinePlotter("label_loss", self.AV, ['label'], [True]))
        self.AVP.append(VisdomLinePlotter("unlabel_loss", self.AV, ['unlabel'], [True]))

    def get_test_dataset(self):
        return DukeSeqLabelDataset(get_path(TestLabelFile), seq_length=LabelSeqLength, data_aug=True,
                                   mean=self.mean, std=self.std)

    def test(self):
        loss = SSWF.test(self)
        self.AV['total'].push_back(loss["total"])
        self.AV['label'].push_back(loss["label"])
        self.AV['unlabel'].push_back(loss["unlabel"])


class TestFolderWF(TestSSWF):  # Type 2
    
    def __init__(self, workingDir, prefix, trained_model, device=None):
        self.acvs = {"label": 20 }
            
        TestSSWF.__init__(self, workingDir, prefix, trained_model, device)

        self.AVP.append(VisdomLinePlotter("label_loss", self.AV, ['label'], [True]))

    def get_test_dataset(self):
        self.testBatch = 50
        return FolderLabelDataset(get_path(TestLabelImgFolder), data_aug=False,
                                  mean=self.mean, std=self.std)

    def calculate_loss(self, val_sample):
        """ label loss only """
        inputImgs = val_sample['img'].to(self.device)
        labels = val_sample['label'].to(self.device)

        output = self.model(inputImgs)
        loss_label = self.criterion(output, labels).item()

        if visualize:
            self.visualize_output(inputImgs, output)
            angle_error, cls_accuracy = angle_metric(
                output.detach().cpu().numpy(), labels.cpu().numpy())
            # print 'label-loss %.4f, angle diff %.4f, accuracy %.4f' % (loss_label, angle_error, cls_accuracy)

        return loss_label

    def test(self):
        loss = SSWF.test(self)
        self.AV['label'].push_back(loss)


class TestUnlabelSeqWF(TestSSWF):  # Type 3

    def __init__(self, workingDir, prefix, trained_model, device=None):
        self.acvs = {"unlabel": 20 }
            
        TestSSWF.__init__(self, workingDir, prefix, trained_model, device)

        self.AVP.append(VisdomLinePlotter("unlabel_loss", self.AV, ['unlabel'], [True]))

    def get_test_dataset(self):
        return FolderUnlabelDataset(get_path(TestUnlabelImgFolder), data_aug=False, include_all=True,
                                    mean=self.mean, std=self.std)

    def calculate_loss(self, val_sample):
        """ unlabel loss only """
        inputImgs = val_sample.squeeze().to(self.device)
        output = self.model(inputImgs)

        loss_unlabel = unlabel_loss_np(output.numpy(), Thresh)

        if self.visualize:
            self.visualize_output(inputImgs, output)
            # print loss_unlabel

        return torch.tensor([loss_unlabel])

    def test(self):
        loss = SSWF.test(self)
        self.AV['unlabel'].push_back(loss)