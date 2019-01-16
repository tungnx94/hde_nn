import torch.nn as nn
import torch.nn.functional as F

from hdeNet import HDENet
from mobileNet import MobileNet_v1

class MobileExtractor(HDENet):

    def __init__(self, hidNum=256, depth_multiplier=0.5, device=None):
        HDENet.__init__(self, device)

        self.base_model = MobileNet_v1(depth_multiplier=depth_multiplier, device=device)
        self.conv7 = nn.Conv2d(hidNum, hidNum, 3)  #conv to 1x1, lower extractor layer

        self.load_to_device()

    def forward(self, x):
        x = self.base_model(x)

        x = self.conv7(x)

        x = F.relu(x, inplace=True)
        x = x.view(x.size()[0], -1)

        return x
