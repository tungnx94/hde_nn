import sys
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np

from workflow import WorkFlow
from dataset.generalData import DataLoader
from network import MobileReg

from utils import loadPretrain, unlabel_loss, angle_metric, seq_show

Lamb = 0.1
Thresh = 0.005  # unlabel_loss threshold

TestBatch = 1
LogParamList = ['Batch', 'SeqLength', 'LearningRate', 'Trainstep',
                'Lamb', 'Thresh']  # these params will be log into the file


class GeneralWF(WorkFlow):

    def __init__(self, workingDir, prefix="", suffix="",
                 device=None, mobile_model=None, trained_model=None):
        super(GeneralWF, self).__init__(workingDir, prefix, suffix)

        self.device = device
        # select default device if not specified
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.lamb = Lamb
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.testBatch = TestBatch
        self.visualize = True

        # Model
        self.model = MobileReg()
        if mobile_model is not None:
            self.model.load_pretrained_pth(mobile_model)

        self.model.to(self.device)

        if trained_model is not None:  # load trained params
            loadPretrain(self.model, trained_model)

        # Test dataset & loader
        self.test_dataset = self.get_test_dataset()
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.testBatch)

        self.criterion = nn.MSELoss()  # loss function

        # Record useful params in logfile
        """
        logstr = ''
        for param in LogParamList:
            logstr += param + ': ' + str(globals()[param]) + ', '
        self.logger.info(logstr)
        """

    def get_test_dataset(self):
        pass

    def visualize_output(self, inputs, outputs):
        seq_show(inputs.cpu().numpy(), dir_seq=outputs.detach().cpu().numpy(),
                 scale=0.8, mean=self.mean, std=self.std)

    def calculate_loss(self, val_sample):
        """ combined loss """
        inputImgs = val_sample['imgseq'].squeeze().to(self.device)
        labels = val_sample['labelseq'].squeeze().to(self.device)

        output = self.model(inputImgs)
        loss_label = self.criterion(output, labels)

        loss_unlabel = unlabel_loss(output.detach().cpu().numpy(), Thresh)
        loss_unlabel = torch.tensor([loss_unlabel]).to(self.device)

        loss_total = loss_label + self.lamb * loss_unlabel

        loss = {"total": loss_total.item(), "label": loss_label.item(),
                "unlabel": loss_unlabel.item()}

        if self.visualize:  # display
            self.visualize_output(inputImgs, output)

            angle_error, cls_accuracy = angle_metric(
                output.detach().cpu().numpy(), labels.cpu().numpy())

            print 'loss: {}, angle diff %.4f, accuracy %.4f'.format(loss, angle_error, cls_accuracy)

        return loss

    def test(self):
        """ test one batch """
        # activate
        super(GeneralWF, self).test()
        self.model.eval()

        # calculate next sample loss
        sample = self.test_loader.next_sample()
        loss = self.calculate_loss(sample)

        return loss

    def run(self):
        pass
