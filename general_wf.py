import torch
import random

import torch.nn as nn
import torch.optim as optim
import numpy as np
import config as cnf

from math import pi
from os.path import join
from torch.utils.data import DataLoader

from general_wf import GeneralWF
from MobileReg import MobileReg
from utils import loadPretrain, seq_show, unlabel_loss, angle_metric

from labelData import LabelDataset
from unlabelData import UnlabelDataset
from folderLabelData import FolderLabelDataset
from folderUnlabelData import FolderUnlabelDataset
from dukeSeqLabelData import DukeSeqLabelDataset

import sys
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

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

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

LogParamList = ['Batch', 'UnlabelBatch', 'learning_rate', 'Trainstep',
                'Lamb', 'Thresh']  # these params will be log into the file


class GeneralWF(Workflow.Workflow):

    def __init__(self, workingDir, prefix="", suffix="", device=None, pre_mobile_model=None, pre_model=None):
        super(General, self).__init__(workingDir, prefix, suffix)

        self.device = device
        # select default device if not specified
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.testBatch = 1
        self.testEpoch = 0
        self.visualize = True

        # Model
        self.model = MobileReg()
        if pre_mobile_model is not None
            self.model.load_pretrained_pth(pre_mobile_model)

        self.model.to(self.device)

        if pre_model is not None  # load trained params
            loadPretrain(self.model, pre_model)

        # Test dataset & loader
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.testBatch, shuffle=True, num_workers=1)

        self.test_data_iter = iter(self.test_loader)

        self.criterion = nn.MSELoss()  # loss function

        # Record useful params in logfile
        logstr = ''
        for param in LogParamList:
            logstr += param + ': ' + str(globals()[param]) + ', '
        self.logger.info(logstr)

    def get_test_dataset(self):
        pass

    def initialize(self):
        """ Initilize """
        super(GeneralWF, self).initialize()
        self.logger.info("Initialized.")

    def visualize_output(self, inputs, outputs):
        seq_show(inputs.cpu().numpy(), dir_seq=outputs.detach().cpu().numpy(),
                 scale=0.8, mean=mean, std=std)

    def next_sample(self, data_iter, loader, epoch):
        """ get next batch, update data_iter and epoch if needed """
        try:
            sample = data_iter.next()
        except:
            data_iter = iter(loader)
            sample = data_iter.next()
            epoch += 1

        return sample, data_iter, epoch

    def calculate_loss(self, val_sample):
        """ combined loss """
        inputImgs = val_sample['imgseq'].squeeze().to(self.device)
        labels = val_sample['labelseq'].squeeze().to(self.device)

        output = self.model(inputImgs)
        loss_label = self.criterion(output, labels)

        loss_unlabel = unlabel_loss(output.numpy(), Thresh)
        loss_unlabel = torch.tensor([loss_unlabel])
        loss = loss_label + Lamb * loss_unlabel

        return {"total": loss.item(), "label": loss_label.item(), "unlabel": loss_unlabel.item()}

    def test(self):
        """ test one batch """
        # activate
        super(GeneralWF, self).test()
        self.model.eval()

        # get next sample
        sample, self.test_data_iter, self.testEpoch = self.next_sample(
            self.test_data_iter, self.test_loader, self.testEpoch)

        # test loss
        loss = self.calculate_loss(sample)
        return loss

    def run():
        pass
