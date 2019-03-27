import math
import torch
import torch.nn as nn

from hdeNet import HDENet

# default parameters
HiddensDF = [1, 8, 16, 16, 32, 32]  # channels
KernelsDF = [4, 4, 3, 4, 2]
PaddingsDF = [1, 1, 1, 1, 0]
StridesDF = [2, 2, 2, 2, 1]

from extractor import BaseExtractor, MobileExtractor


class EncoderReg_Pred(HDENet):

    def __init__(self, hiddens=HiddensDF, kernels=KernelsDF, strides=StridesDF, paddings=PaddingsDF, actfunc='relu', regnum=2, rnnHidNum=128, device=None):
        HDENet.__init__(self, device)

        self.codenum = hiddens[-1]  # input for LSTM (features), should be 256
        self.rnnHidNum = rnnHidNum  # hidden layer size for LSTMs

        self.encoder = BaseExtractor(
            hiddens, kernels, strides, paddings, actfunc, device=device)  # can be replaced using MobileNet

        # self.encoder = MobileExtractor(depth_multiplier=0.5)

        self.reg = nn.Linear(self.codenum, regnum)

        # input_size, hidden_size respectively
        self.pred_en = nn.LSTM(self.codenum, rnnHidNum)

        self.pred_de = nn.LSTM(self.codenum, rnnHidNum)
        self.pred_de_linear = nn.Linear(self.rnnHidNum, self.codenum)  # FC

        self._initialize_weights()
        self.load_to_device()

    def init_hidden(self, hidden_size, batch_size=1):
        h1 = self.new_variable(torch.zeros(
            1, batch_size, hidden_size))  # hidden state
        h2 = self.new_variable(torch.zeros(
            1, batch_size, hidden_size))  # cell state

        return (h1, h2)

    def forward(self, x):
        seq_length = x_encode.size()[0]  # Seq

        x_encode = self.encoder(x)  # features
        x_encode = x_encode.view(seq_length, -1)  # 2d : Seq x Features

        x_reg = self.reg(x_encode)  # (sine, cosine)

        # predictor: use first half as input, last half as target (good ?)
        innum = seq_length / 2

        # input of LSTM is [SeqLength x Batch x InputSize], SeqLength variable
        pred_in = x_encode[:innum].unsqueeze(1)  # add batch=1 dimension
        hidden_0 = self.init_hidden(self.rnnHidNum, 1)  # batch=1

        # output = [SeqLength x Batch x HiddenSize], (hidden_n, cell_n)
        pred_en_out, hidden = self.pred_en(pred_in, hidden_0)

        pred_de_in = self.new_variable(torch.zeros(1, 1, self.codenum))
        # could use pred_de_in = x_encode[innum-1].unsqueeze(0).unsqueeze(0) ?

        pred_out = []
        for k in range(innum, seq_length):  # input the decoder one by one cause there's a loop
            pred_de_out, hidden = self.pred_de(pred_de_in, hidden)

            pred_de_out = self.pred_de_linear(
                pred_de_out.view(1, self.rnnHidNum))
            pred_de_in = pred_de_out.detach().unsqueeze(1)

            pred_out.append(pred_de_out)

        pred_out = torch.cat(tuple(pred_out), dim=0)

        return x_reg, x_encode, pred_out

if __name__ == '__main__':  # test
    import sys
    sys.path.insert(0, "..")
    from utils import get_path
    from dataset import DataLoader, SequenceUnlabelDataset

    hiddens = [3, 16, 32, 32, 64, 64, 128, 256]
    kernels = [4, 4, 4, 4, 4, 4, 3]
    paddings = [1, 1, 1, 1, 1, 1, 0]
    strides = [2, 2, 2, 2, 2, 2, 1]

    seq_length = 16
    lr = 0.005
    stateEncoder = EncoderReg_Pred(
        hiddens, kernels, strides, paddings, actfunc='leaky', rnnHidNum=128)
    criterion = nn.MSELoss()

    print stateEncoder
    # data
    imgdataset = SequenceUnlabelDataset("train", path=get_path(
        "UCF"), seq_length=seq_length, data_aug=True)
    dataloader = DataLoader(imgdataset)  # batch_size = 1

    for ind in range(10):
        sample = dataloader.next_sample()
        inputVar = stateEncoder.new_variable(sample.squeeze())

        x, encode, pred = stateEncoder(inputVar)

        pred_target = encode[seq_length / 2:, :].detach()
        loss_pred = criterion(pred, pred_target)  # unlabel

        print ind, loss_pred.item()
