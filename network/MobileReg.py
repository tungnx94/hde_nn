import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from hdenet import HDENet
from mobilenet import mobilenet_v1

class MobileReg(HDENet):

    def __init__(self, hidnum=256, regnum=2, lamb=0.1, thresh=0.005, device=None):  
        # input size should be 112 ?
        HDENet.__init__(self, device=device)

        self.feature = mobilenet_v1(0.50)
        self.conv7 = nn.Conv2d(hidnum, hidnum, 3)  #conv to 1 by 1
        self.reg = nn.Linear(hidnum, regnum)

        self.criterion = nn.MSELoss()  # L2 loss
        self.lamb = lamb
        self.thresh = thresh
        self._initialize_weights()

    def load_mobilenet(self, fname):
        self.feature.load_from_npz(fname)

    def forward(self, x):
        x = x.to(self.device)

        x = self.feature(x)
        x = F.relu(self.conv7(x), inplace=True)
        x = self.reg(x.view(x.size()[0], -1))

        return x

    def unlabel_loss(self, output, threshold):
        """
        :param output: network unlabel output tensor
        :return: unlabel loss tensor
        """
        output = output.to(self.device)
        unlabel_batch = output.shape[0]
        loss = torch.Tensor([0]).to(self.device).float()
        threshold = torch.tensor(threshold).to(self.device).float()

        for ind1 in range(unlabel_batch - 5):  # try to make every sample contribute
            # randomly pick two other samples
            ind2 = random.randint(ind1 + 2, unlabel_batch - 1)  # big distance
            ind3 = random.randint(ind1 + 1, ind2 - 1)  # small distance

            diff_big = torch.sum(
                (output[ind1] - output[ind2]) ** 2).float() / 2.0
            diff_small = torch.sum(
                (output[ind1] - output[ind3]) ** 2).float() / 2.0

            cost = torch.max(diff_small - diff_big - threshold,
                             torch.tensor(0).to(self.device).float())
            loss += cost

        return loss

    def forward_unlabel(self, inputs):
        """
        :param sample: unlabeled data
        :return: unlabel loss
        """
        inputs = inputs.to(self.device)
        outputs = self(inputs)

        loss = self.unlabel_loss(outputs, self.thresh)
        return loss.to(self.device).float()

    def forward_label(self, inputs, targets):
        """
        :param sample: labeled data
        :return: label loss
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self(inputs)
        return self.criterion(outputs, targets)

    def forward_combine(self, inputs, targets, inputs_unlabel):
        loss_label = self.forward_label(inputs, targets)
        loss_unlabel = self.forward_unlabel(inputs_unlabel)
        loss_total = loss_label + self.lamb*loss_unlabel

        loss = {"total": loss_total,
                "label": loss_label,
                "unlabel": loss_unlabel}

        return loss

    def _initialize_weights(self):
        for m in self.modules():
            # print type(m)
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


def main():
    import torch
    from torch.autograd import Variable

    inputVar = Variable(torch.rand((10, 3, 192, 192)))

    net = MobileReg()
    net.load_mobilenet('pretrained_models/mobilenet_v1_0.50_224.pth')

    outputVar = net(inputVar)
    print outputVar

if __name__ == '__main__':
    main()
