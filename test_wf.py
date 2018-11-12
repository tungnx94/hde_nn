import sys
import torch
import random

import torch.nn as nn
import torch.optim as optim
import numpy as np
import config as cnf

from math import pi
from os.path import join
from torch.utils.data import DataLoader

from workflow import WorkFlow
from MobileReg import MobileReg
from utils import loadPretrain, seq_show, unlabel_loss, angle_metric

from labelData import LabelDataset
from unlabelData import UnlabelDataset
from folderLabelData import FolderLabelDataset
from folderUnlabelData import FolderUnlabelDataset
from dukeSeqLabelData import DukeSeqLabelDataset

sys.path.append('../WorkFlow')

exp_prefix = 'vis_1_3_'  # ?
Batch = 128
UnlabelBatch = 24  # 32
learning_rate = 0.0005  # learning rate
Trainstep = 20000  # number of train() calls
Lamb = 0.1  # ?
Thresh = 0.005  # unlabel_loss threshold
TestBatch = 1

Snapshot = 5000  # do a snapshot every Snapshot steps (save period)
TestIter = 10  # do a testing every TestIter steps
ShowIter = 1  # print to screen

# hardcode in labelData, used where ?
train_label_file = '/datadrive/person/DukeMTMC/trainval_duke.txt'
test_label_file = '/datadrive/person/DukeMTMC/test_heading_gt.txt'
unlabel_file = 'duke_unlabeldata.pkl'
saveModelName = 'facing'

test_label_img_folder = '/home/wenshan/headingdata/val_drone'
test_unlabel_img_folder = '/datadrive/exp_bags/20180811_gascola'

pre_mobile_model = 'pretrained_models/mobilenet_v1_0.50_224.pth'
load_pre_mobile = False

pre_model = 'models/1_2_facing_20000.pkl'
load_pre_train = True

TestType = 2  # 0: none, 1: labeled sequence, 2: labeled folder, 3: unlabeled sequence
LogParamList = ['Batch', 'UnlabelBatch', 'learning_rate', 'Trainstep',
                'Lamb', 'Thresh']  # these params will be log into the file


class TestWF(GeneralWF):

    def run():
        for iteration in range(Trainstep):
            self.test()
        print "Finished testing"


class TestLabelSeqWF(TestWF):  # Type 1

    def get_test_dataset()
        return DukeSeqLabelDataset(labelfile=test_label_file, batch=UnlabelBatch, data_aug=True, mean=mean, std=std)


class TestFolderWF(TestWF):  # Type 2

    def get_test_dataset():
        self.testBatch = 50
        return FolderLabelDataset(imgdir=test_label_img_folder, data_aug=False, mean=mean, std=std)

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
            print 'label-loss %.4f, angle diff %.4f, accuracy %.4f' % (loss_label, angle_error, cls_accuracy)

        return loss_label


class TestUnlabelSeqWF(TestWF):  # Type 3

    def get_test_dataset():
        return FolderUnlabelDataset(imgdir=test_unlabel_img_folder, data_aug=False, include_all=True, mean=mean, std=std)

    def calculate_loss(self, val_sample):
        """ unlabel loss only """
        inputImgs = val_sample.squeeze().to(self.device)
        output = self.model(inputImgs)

        loss_unlabel = unlabel_loss(output.numpy(), Thresh)

        # import ipdb;ipdb.set_trace()
        if visualize:
            self.visualize_output(inputImgs, output)
            print loss_unlabel

        return torch.tensor([loss_unlabel])
