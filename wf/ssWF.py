import sys
sys.path.append("..")

import torch
import torch.nn as nn

from workflow import WorkFlow
from dataset import DataLoader
from network import MobileReg

from utils import unlabel_loss_np, angle_metric, seq_show

Lamb = 0.1
Thresh = 0.005  # unlabel_loss threshold
TestBatch = 1


class SSWF(WorkFlow):

    def __init__(self, mobile_model=None):
        self.lamb = Lamb
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.visualize = False
        self.mobile_model = mobile_model

        # Test dataset & loader
        self.test_dataset = self.get_test_dataset()
        self.test_loader = DataLoader(self.test_dataset, batch_size=TestBatch)

        self.criterion = nn.MSELoss()  # loss function

    def load_model(self):
        model = MobileReg()
        if self.mobile_model is not None:
            model.load_mobilenet(self.mobile_model) 
            self.logger.info("Loaded MobileNet model: {}".format(self.mobile_model))

        return model

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

        loss_unlabel = unlabel_loss_np(output.detach().cpu().numpy(), Thresh)
        loss_unlabel = torch.tensor([loss_unlabel]).to(self.device).float()

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
        WorkFlow.test(self)
        self.model.eval() # activate

        # calculate next sample loss
        sample = self.test_loader.next_sample()
        loss = self.calculate_loss(sample)

        return loss
    