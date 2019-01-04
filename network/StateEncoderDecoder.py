import sys
sys.path.insert(0, "..")

import math
import torch
import torch.nn as nn

from hdenet import HDENet

# default network parameters
HiddensDF = [1, 8, 16, 16, 32, 32]  # 14, 7, 4, 2, 1
KernelsDF = [4, 4, 3, 4, 2]
PaddingsDF = [1, 1, 1, 1, 0]
StridesDF = [2, 2, 2, 2, 1]


class StateCoder(HDENet):
    """ 
    deep ConvNet
    can be used as encoder or decoder
    """

    def __init__(self, hiddens, kernels, strides, paddings, actfunc, device=None):
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


class EncoderReg_Pred(HDENet):

    def __init__(self, hiddens=HiddensDF, kernels=KernelsDF, strides=StridesDF, paddings=PaddingsDF, actfunc='relu', regnum=2, rnnHidNum=128, device=None):
        HDENet.__init__(self, device)

        self.codenum = hiddens[-1]  # input size for LSTM
        self.rnnHidNum = rnnHidNum  # hidden layer size

        self.encoder = StateCoder(
            hiddens, kernels, strides, paddings, actfunc, device=device)
        self.encoder.load_to_device()

        self.reg = nn.Linear(self.codenum, regnum)

        self.pred_en = nn.LSTM(self.codenum, rnnHidNum)

        self.pred_de = nn.LSTM(self.codenum, rnnHidNum)
        self.pred_de_linear = nn.Linear(self.rnnHidNum, self.codenum)  # FC

    def init_hidden(self, hidden_size, batch_size=1):
        h1 = self.new_variable(torch.zeros(
            1, batch_size, hidden_size))  # hidden state
        h2 = self.new_variable(torch.zeros(
            1, batch_size, hidden_size))  # cell state

        return (h1, h2)

    def forward(self, x):
        x_encode = self.encoder(x)
        seq_length = x_encode.size()[0]

        x_encode = x_encode.view(seq_length, -1)

        # regression (sin, cosin)
        x_reg = self.reg(x_encode)

        # rnn predictor
        # use first half as input, last half as target (why though ?)
        innum = seq_length / 2

        # input of LSTM is [SeqLength x Batch x InputSize] with SeqLength
        # varible
        pred_in = x_encode[:innum].unsqueeze(1)  # add batch dimension (=1)
        hidden = self.init_hidden(self.rnnHidNum, 1)  # batch = 1

        # output = [SeqLength x Batch x HiddenSize]
        pred_en_out, hidden = self.pred_en(pred_in, hidden)

        pred_de_in = self.new_variable(torch.zeros(1, 1, self.codenum))

        pred_out = []
        for k in range(innum, seq_length):  # input the decoder one by one cause there's a loop
            pred_de_out, hidden = self.pred_de(pred_de_in, hidden)

            pred_de_out = self.pred_de_linear(
                pred_de_out.view(1, self.rnnHidNum))
            pred_de_in = pred_de_out.detach().unsqueeze(1)

            pred_out.append(pred_de_out)

        pred_out = torch.cat(tuple(pred_out), dim=0)

        return x_reg, x_encode, pred_out

if __name__ == '__main__':

    import torch.optim as optim
    import matplotlib.pyplot as plt

    from utils import get_path
    from dataset import DataLoader, FolderUnlabelDataset

    hiddens = [3, 16, 32, 32, 64, 64, 128, 256]
    kernels = [4, 4, 4, 4, 4, 4, 3]
    paddings = [1, 1, 1, 1, 1, 1, 0]
    strides = [2, 2, 2, 2, 2, 2, 1]

    seq_length = 16
    lr = 0.005

    stateEncoder = EncoderReg_Pred(
        hiddens, kernels, strides, paddings, actfunc='leaky', rnnHidNum=128)
    stateEncoder.load_to_device()


    paramlist = list(stateEncoder.parameters())

    print stateEncoder
    print len(paramlist)

    # data
    imgdataset = FolderUnlabelDataset("train", img_dir=get_path(
        "dirimg"), seq_length=seq_length, data_aug=True, include_all=True)
    dataloader = DataLoader(imgdataset)  # batch_size = 1

    criterion = nn.MSELoss()
    regOptimizer = optim.SGD(stateEncoder.parameters(), lr=lr, momentum=0.9)
    # regOptimizer = optim.Adam(stateEncoder.parameters(), lr = lr)

    lossplot = []
    encodesumplot = []

    ind = 200
    for sample in dataloader:
        inputVar = stateEncoder.new_variable(sample.squeeze())

        x, encode, pred = stateEncoder(inputVar)

        pred_target = encode[seq_length / 2:, :].detach()
        loss_pred = criterion(pred, pred_target)  # unlabel

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
