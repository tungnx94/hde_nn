import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from mobilenet import mobilenet_v1


class MobileReg(nn.Module):

    def __init__(self, hidnum=256, regnum=2):  
        # input size should be 112

        super(MobileReg, self).__init__()

        self.feature = mobilenet_v1(0.50)
        self.conv7 = nn.Conv2d(hidnum, hidnum, 3)  #conv to 1 by 1
        self.reg = nn.Linear(hidnum, regnum)
        
        self._initialize_weights()

    def forward(self, x):
        #import ipdb; ipdb.set_trace()
        
        x = self.feature(x)
        # print x.size()

        x = F.relu(self.conv7(x), inplace=True)
        # print x.size()

        x = self.reg(x.view(x.size()[0], -1))

        return x

    def _initialize_weights(self):
        for m in self.modules():
            print type(m)
            if isinstance(m, nn.Conv2d):
                # print 'conv2d'
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # print 'batchnorm'
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # print 'linear'
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load_pretrained_pth(self, fname):  
        """ load mobilenet-from-tf - amigo """
        params = torch.load(fname)
        self.feature.load_from_npz(params)


def main():
    inputVar = Variable(torch.rand((10, 3, 192, 192)))

    net = MobileReg()
    net.load_pretrained_pth('pretrained_models/mobilenet_v1_0.50_224.pth')

    outputVar = net(inputVar)
    
    print outputVar

if __name__ == '__main__':
    main()
