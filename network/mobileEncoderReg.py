# based on MobileReg and StateEncoderDecoder
import sys
sys.path.insert(0, "..")

import torch.nn as nn
import torch.nn.functional as F
 
from mobileReg import MobileReg

class MobileEncoderReg(MobileReg):

    def __init__(self, hidNum=256, rnnHidNum=128, regNum=2, lamb=0.1, device=None):
        # input tensor should be [Batch, 3, 192, 192]
        self.rnnHidNum = rnnHidNum

        self.pred_en = nn.LSTM(hidNum, rnnHidNum)
        self.pred_de = nn.LSTM(hidNum, rnnHidNum)
        self.pred_de_linear = nn.Linear(rnnHidNum, hidNum)  # FC

        MobileReg.__init__(self, hidnum=hidNum, regnum=regNum, device=device)

    def unlabel_loss(self, output):
        pass

    def init_hidden(self, hidden_size, batch_size=1):
        h1 = self.new_variable(torch.zeros(
            1, batch_size, hidden_size))  # hidden state
        h2 = self.new_variable(torch.zeros(
            1, batch_size, hidden_size))  # cell state

        return (h1, h2)

    def forward_unlabel(self, inputs):
        # input size is [SeqLength, 3, W, H]
        x_encode = self.extract_features(inputs)
        seq_length = x_encode.size()[0] # SeqLength

        # prediction: use first half as input, last half as target
        innum = seq_length / 2

        # input of LSTM is [SeqLength x Batch x InputSize]
        # output = [SeqLength x Batch x HiddenSize], (hidden_n, cell_n)
        pred_in = x_encode[:innum].unsqueeze(1)  # add batch=1 dimension
        hidden_0 = self.init_hidden(self.rnnHidNum, 1)  # batch=1

        pred_out = []
        pred_en_out, hidden = self.pred_en(pred_in, hidden_0)
        pred_de_in = self.new_variable(torch.zeros(1, 1, self.codenum))
        # could use pred_de_in = x_encode[innum-1].unsqueeze(0).unsqueeze(0) ? 

        for k in range(innum, seq_length):  # decode one by one
            pred_de_out, hidden = self.pred_de(pred_de_in, hidden)

            pred_de_out = self.pred_de_linear(pred_de_out.view(1, self.rnnHidNum))
            pred_de_in = pred_de_out.detach().unsqueeze(1)

            pred_out.append(pred_de_out)

        pred_out = torch.cat(tuple(pred_out), dim=0)

        loss = self.criterion(pred, pred_target)
        return loss

    def _initialize_weights(self):
        # init weights for all submodules
        MobileReg._initialize_weights(self)

        nn.init.xavier_normal(self.pred_en)
        nn.init.xavier_normal(self.pred_de)


if __name__ == "__main__": # test
    from dataset import TrackingLabelDataset, FolderUnlabelDataset, DataLoader
    # prepare data
    imgdataset = TrackingLabelDataset("duke-train",
                                      data_file=DukeLabelFile, data_aug=True)
    unlabelset = FolderUnlabelDataset("ucf-train", dat_file="../data/ucf_unlabeldata.pkl",
                                      seq_length=UnlabelBatch, data_aug=True, extend=True)
    dataloader = DataLoader(imgdataset, batch_size=20)
    unlabelloader = DataLoader(unlabelset)

    model = MobileEncoderReg()
    model.load_mobilenet('pretrained_models/mobilenet_v1_0.50_224.pth')

    val_loss = 0.0
    for ind in range(1, 50):
        sample = dataloader.next_sample()
        sample_unlabel = unlabelloader.next_sample()
