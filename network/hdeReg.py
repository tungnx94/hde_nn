### Vanilla model CNN
import torch
import torch.nn as nn

from .hdeNet import HDENet
from .extractor import BaseExtractor, MobileExtractor

SizeDf = [192, 64, 32, 16, 5, 3, 1]
HiddensDF = [3, 32, 64, 64, 128, 128, 256]  # channels
KernelsDF = [5, 5, 5, 3, 3, 3]
PaddingsDF = [1, 2, 2, 1, 1, 0]
StridesDF = [3, 2, 2, 3, 2, 1]


class HDEReg(HDENet):

    def __init__(self, extractor, hidNum=256, device=None, init=True):
        # input size should be [192x192]
        HDENet.__init__(self, device)

        if extractor == "base":
            self.feature = BaseExtractor(hiddens=HiddensDF, kernels=KernelsDF, strides=StridesDF, paddings=PaddingsDF)
        else:
            self.feature = MobileExtractor(hidNum, depth_multiplier=0.5, device=device)    # reinited, could be better, default 0.5

        self.criterion = nn.MSELoss(reduction='none')  # L2    
        self.reg = nn.Sequential(
            nn.Linear(hidNum, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        if init:
            self._initialize_weights()
            self.load_to_device()
            
    def load_mobilenet(self, fname):
        self.feature.load_from_npz(fname)

    def forward(self, x):
        batch = x.shape[0]
        x = x.to(self.device)
        # x = self.feature(x).squeeze() # get rid of the 1-dim
        x = self.feature(x).view(batch, -1)
        x = self.reg(x)
        return x

    def loss_label(self, inputs, targets, mean=False):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        if mean:
            loss = torch.mean(loss)
        return loss
