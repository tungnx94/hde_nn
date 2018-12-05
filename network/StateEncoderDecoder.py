import os
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

# these are default values of the network
# if the initial values are not assigned, these values will be used
HiddensDF = [1, 8, 16, 16, 32, 32]  # 14, 7, 4, 2, 1
KernelsDF = [4, 4, 3, 4, 2]
PaddingsDF = [1, 1, 1, 1, 0]
StridesDF = [2, 2, 2, 2, 1]


class StateCoder(nn.Module):
    """ 
    deep ConvNet
    can be used as encoder or decoder
    """

    def __init__(self, hiddens, kernels, strides, paddings, actfunc):
        super(StateCoder, self).__init__()

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

        self._initialize_weights()

    def forward(self, x):

        return self.coder(x)

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


class StateEncoderDecoder(nn.Module):

    def __init__(self, hiddens=HiddensDF, kernels=KernelsDF, strides=StridesDF, paddings=PaddingsDF, actfunc='relu'):
        super(StateEncoderDecoder, self).__init__()
        self.encoder = StateCoder(
            hiddens, kernels, strides, paddings, actfunc)

        self.decoder = StateCoder(
            hiddens[::-1], kernels[::-1], strides[::-1], paddings[::-1], actfunc)

    def forward(self, x):
        x_encode = self.encoder(x)
        x = self.decoder(x_encode)
        return x, x_encode

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


class EncoderFC(nn.Module):
    """ encoder + fc-layer """

    def __init__(self, hiddens, kernels, strides, paddings, actfunc='relu', fc_layer=2):
        super(EncoderFC, self).__init__()
        self.encoder = StateCoder(
            hiddens, kernels, strides, paddings, actfunc)

        self.fc = nn.Linear(hiddens[-1], fc_layer)

    def forward(self, x):
        x_encode = self.encoder(x)
        x = self.fc(x_encode.view(x_encode.size()[0], -1))
        return x, x_encode


class EncoderCls(EncoderFC):
    """ encoder classificator """

    def __init__(self, hiddens=HiddensDF, kernels=KernelsDF, strides=StridesDF, paddings=PaddingsDF, actfunc='relu', clsnum=8):
        super(EncoderCls, self).__init__(hiddens, kernels,
                                         strides, paddings, actfunc, fc_layer=clsnum)


class EncoderReg(EncoderFC):
    """ encoder regressor """

    def __init__(self, hiddens=HiddensDF, kernels=KernelsDF, strides=StridesDF, paddings=PaddingsDF, actfunc='relu', regnum=2):
        super(EncoderCls, self).__init__(hiddens, kernels,
                                         strides, paddings, actfunc, fc_layer=regnum)


class EncoderReg_norm(EncoderReg):
    """ normalized version of EncoderReg """

    def forward(self, x):
        x_encode = self.encoder(x)
        x = self.fc(x_encode.view(x_encode.size()[0], -1))
        y = x.abs()  # normalize so |x| + |y| = 1
        y = y.sum(dim=1)
        # import ipdb; ipdb.set_trace()
        x = x / y.unsqueeze(1)
        return x, x_encode


class EncoderReg_Pred(nn.Module):

    def __init__(self, hiddens=HiddensDF, kernels=KernelsDF, strides=StridesDF, paddings=PaddingsDF, actfunc='relu', regnum=2, rnnHidNum=128):
        super(EncoderReg_Pred, self).__init__()
        self.encoder = StateCoder(
            hiddens, kernels, strides, paddings, actfunc)
        self.reg = nn.Linear(hiddens[-1], regnum)
        self.codenum = hiddens[-1]

        self.rnnHidNum = rnnHidNum
        self.pred_en = nn.LSTM(hiddens[-1], rnnHidNum)
        self.pred_de = nn.LSTM(hiddens[-1], rnnHidNum)
        self.pred_de_linear = nn.Linear(self.rnnHidNum, self.codenum)

    def init_hidden(self, hidden_size, batch_size):
        return (Variable(torch.zeros(1, batch_size, hidden_size)).cuda(),
                Variable(torch.zeros(1, batch_size, hidden_size)).cuda())

    def forward(self, x):
        x_encode = self.encoder(x)
        batchsize = x_encode.size()[0]
        x_encode = x_encode.view(batchsize, -1)
        # regression the direction
        x = self.reg(x_encode)
        # y = x.abs() # normalize so |x| + |y| = 1
        # y = y.sum(dim=1)
        # x = x/y.unsqueeze(1)

        # rnn predictor
        innum = batchsize / 2  # use first half as input, last half as target
        # input of LSTM should be T x batch x InLen
        pred_in = x_encode[0:innum, :].unsqueeze(1)
        hidden = self.init_hidden(self.rnnHidNum, 1)
        pred_en_out, hidden = self.pred_en(pred_in, hidden)

        # import ipdb; ipdb.set_trace()
        pred_de_in = Variable(torch.zeros(1, 1, self.codenum)).cuda()
        pred_out = []
        for k in range(innum, batchsize):  # input the decoder one by one cause there's a loop
            pred_de_out, hidden = self.pred_de(pred_de_in, hidden)
            pred_de_out = self.pred_de_linear(
                pred_de_out.view(1, self.rnnHidNum))
            pred_out.append(pred_de_out)
            pred_de_in = pred_de_out.detach().unsqueeze(1)

        pred_out = torch.cat(tuple(pred_out), dim=0)
        return x, x_encode, pred_out

if __name__ == '__main__':

    import torch.optim as optim
    import matplotlib.pyplot as plt

    from dataset import DataLoader, FolderUnlabelDataset
    from utils.data import get_path

    hiddens = [3, 16, 32, 32, 64, 64, 128, 256]
    kernels = [4, 4, 4, 4, 4, 4, 3]
    paddings = [1, 1, 1, 1, 1, 1, 0]
    strides = [2, 2, 2, 2, 2, 2, 1]

    unlabel_batch = 4
    lr = 0.005

    stateEncoder = EncoderReg_Pred(
        hiddens, kernels, strides, paddings, actfunc='leaky', rnnHidNum=128)
    if torch.cuda.is_available():
        stateEncoder.cuda()

    paramlist = list(stateEncoder.parameters())

    print stateEncoder
    print len(paramlist)

    # data
    imgdataset = FolderUnlabelDataset(img_dir=get_path(
        "dirimg"), seq_length=unlabel_batch, data_aug=True, include_all=True)
    dataloader = DataLoader(imgdataset)

    criterion = nn.MSELoss()
    regOptimizer = optim.SGD(stateEncoder.parameters(), lr=lr, momentum=0.9)
    # regOptimizer = optim.Adam(stateEncoder.parameters(), lr = lr)

    lossplot = []
    encodesumplot = []

    ind = 200
    for sample in dataloader:
        inputVar = Variable(sample.squeeze()).cuda()
        x, encode, pred = stateEncoder(inputVar)

        """
        print inputVar.size()
        print encode
        print encode.size(), x.size(), pred.size()

        # loss = loss_label + loss_pred * lamb #+ normloss * lamb2
        # loss.backward()
        """

        pred_target = encode[unlabel_batch / 2:, :].detach()
        loss_pred = criterion(pred, pred_target)

        # back propagate
        regOptimizer.zero_grad()
        loss_pred.backward()
        regOptimizer.step()

        lossplot.append(loss_pred.item())
        encodesumplot.append(encode.mean().item())
        print ind, loss_pred.item(), encode.mean().item()

        ind -= 1
        if ind < 0:
            break

    # plot data
    plt.plot(lossplot)
    plt.plot(encodesumplot)
    plt.grid()
    plt.show()
