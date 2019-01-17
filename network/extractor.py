import torch.nn as nn
import torch.nn.functional as F

from hdeNet import HDENet
from mobileNet import MobileNet_v1


class MobileExtractor(HDENet):

    def __init__(self, hidNum=256, depth_multiplier=0.5, device=None):
        HDENet.__init__(self, device)

        self.base_model = MobileNet_v1(
            depth_multiplier=depth_multiplier, device=device)
        # conv to 1x1, lower extractor layer
        self.conv7 = nn.Conv2d(hidNum, hidNum, 3)

        self.load_to_device()

    def forward(self, x):
        x = self.base_model(x)
        x = self.conv7(x)

        x = F.relu(x, inplace=True)
        x = x.view(x.size()[0], -1)
        return x
        
SizeDf = [96, 32, 10, 4, 1]
HiddensDF = [3, 32, 64, 64, 256]  # channels
KernelsDF = [5, 5, 3, 4]
PaddingsDF = [1, 0, 1, 0]
StridesDF = [3, 3, 3, 1]


class BaseExtractor(HDENet):
    """ deep ConvNet, used as en/decoder """

    def __init__(self, hiddens=HiddensDF, kernels=KernelsDF, strides=StridesDF, paddings=PaddingsDF,
                 actfunc="relu", device=None):
        HDENet.__init__(self, device)

        self.coder = nn.Sequential()
        for k in range(len(hiddens) - 1):
            # add conv layer
            conv = nn.Conv2d(hiddens[k], hiddens[k + 1], kernels[k],
                             stride=strides[k], padding=paddings[k])
            self.coder.add_module('conv%d' % (k + 1), conv)

            # add activation layer
            if actfunc == 'leaky':
                self.coder.add_module('relu%d' % (
                    k + 1), nn.LeakyReLU(0.1, inplace=True))
            else:
                self.coder.add_module('relu%d' %
                                      (k + 1), nn.ReLU(inplace=True))

        self.load_to_device()

    def forward(self, x):
        x = self.coder(x)
        return x
