import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models 

from hdenet import HDENet
from mobilenet import MobileNet_v1

MobileNetPretrained = 'network/pretrained_models/mobilenet_v1_0.50_224.pth'

class GRUBaseline(HDENet):

    def __init__(self, init_weights=True):
        super(GRUBaseline, self).__init__()
        self.base_model = models.vgg16().features
        #self.base_model = mobilenet_v1_050()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )

        self.gru = nn.GRU(4096, 1, 2)

        if init_weights():
            self._initialize_weights()

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        #ipdb.set_trace()
        h0 = torch.randn(2, 1)
        output, hn = self.gru(x, h0)

        return x # not output ?

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class MobilenetGRU(HDENet):

    def __init__(self, init_weights=True, hidnum=256, regnum=2,
                 pretrain_model=MobileNetPretrained,
                 batch_size=1):
        super(MobilenetGRU, self).__init__()
        self.batch_size = batch_size

        # self.base_model = models.vgg16().features
        self.base_model = MobileNet_v1(depth_multiplier=0.5, device=device) 

        self.conv7 = nn.Conv2d(hidnum, hidnum, 3)  # conv to 1 by 1

        self.reg = nn.Linear(hidnum, regnum)

        # self.gru = nn.GRU(256, 2, 2)
        self.gru = nn.GRU(256, 5, 2)

        self.logit = nn.Linear(5, 2)
        self.logit2 = nn.Linear(256, 2)
        self.pool2d = nn.AvgPool2d(3)
        
        if init_weights:
            self._initialize_weights()
            self.load_pretrained_pth(pretrain_model)

    def load_pretrained_pth(self, fname):  # load mobilenet-from-tf - amigo
        self.base_model.load_from_npz(fname)

    def forward(self, x):
        # ipdb.set_trace()
        x = self.base_model(x)
        #x = self.pool2d(x)
        x = F.relu(self.conv7(x), inplace=True)
        x = x.view(x.size(0), self.batch_size, -1)

        x, hn = self.gru(x)
        x = x.view(x.size(0), -1)
        #x = F.softmax(self.logit(x))
        output = self.logit(x)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
