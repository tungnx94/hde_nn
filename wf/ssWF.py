import sys
sys.path.append("..")

import torch
import torch.nn as nn

from workflow import WorkFlow
from dataset import DataLoader
from network import MobileReg

from utils import seq_show

Lamb = 0.1
Thresh = 0.005  # unlabel_loss threshold
TestBatch = 1

device = None

class SSWF(WorkFlow):

    def __init__(self, mobile_model=None):
        self.lamb = Lamb
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.mobile_model = mobile_model

        # Test dataset & loader
        self.test_dataset = self.get_test_dataset()
        self.test_loader = DataLoader(self.test_dataset, batch_size=TestBatch)

    def load_model(self):
        model = MobileReg(lamb=Lamb, thresh=Thresh)
        if self.mobile_model is not None:
            model.load_mobilenet(self.mobile_model) 
            self.logger.info("Loaded MobileNet model: {}".format(self.mobile_model))

        return model

    def visualize_output(self, inputs, outputs):
        seq_show(inputs.cpu().numpy(), dir_seq=outputs.detach().cpu().numpy(),
                 scale=0.8, mean=self.mean, std=self.std)

    def get_test_dataset(self):
        pass

    def calculate_loss(self, val_sample):
        pass

    def test(self):
        """ test one batch """
        WorkFlow.test(self)
        self.model.eval() # activate

        # calculate next sample loss
        sample = self.test_loader.next_sample()
        return self.calculate_loss(sample)
    