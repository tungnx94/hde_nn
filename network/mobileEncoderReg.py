# based on MobileReg and StateEncoderDecoder
import sys
sys.path.insert(0, "..")

import torch
import torch.nn as nn

from hdeNet import HDENet
from mobileReg import MobileReg
from mobileNet import MobileNet_v1

class MobileEncoderReg(MobileReg):

    def __init__(self, hidNum=256, rnnHidNum=128, regNum=2, lamb=0.001, device=None):
        # input tensor should be [Batch, 3, 192, 192]
        HDENet.__init__(self, device=device)

        self.lamb = lamb
        self.hidNum = hidNum
        self.rnnHidNum = rnnHidNum
        self.criterion = nn.MSELoss()

        self.feature = MobileNet_v1(depth_multiplier=0.5, device=device) # feature extractor, upper layers
        self.conv7 = nn.Conv2d(hidNum, hidNum, 3)  #conv to 1x1, lower extractor layer
        self.reg = nn.Linear(hidNum, regNum) # regression (sine, cosine)

        self.pred_en = nn.LSTM(hidNum, rnnHidNum)
        self.pred_de = nn.LSTM(hidNum, rnnHidNum)
        self.pred_de_linear = nn.Linear(rnnHidNum, hidNum)  # FC

        self._initialize_weights()
        self.load_to_device()

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
        seq_length = x_encode.size()[0]  # SeqLength

        # prediction: use first half as input, last half as target
        innum = seq_length / 2

        # input of LSTM is [SeqLength x Batch x InputSize]
        # output = [SeqLength x Batch x HiddenSize], (hidden_n, cell_n)
        pred_in = x_encode[:innum].unsqueeze(1)  # add batch=1 dimension
        hidden_0 = self.init_hidden(self.rnnHidNum, 1)  # batch=1

        pred_out = []
        pred_en_out, hidden = self.pred_en(pred_in, hidden_0)
        pred_de_in = self.new_variable(torch.zeros(1, 1, self.hidNum))
        # could use pred_de_in = x_encode[innum-1].unsqueeze(0).unsqueeze(0) ?

        for k in range(innum, seq_length):  # decode one by one
            pred_de_out, hidden = self.pred_de(pred_de_in, hidden)

            pred_de_out = self.pred_de_linear(
                pred_de_out.view(1, self.rnnHidNum))
            pred_de_in = pred_de_out.detach().unsqueeze(1)

            pred_out.append(pred_de_out)

        pred_out = torch.cat(tuple(pred_out), dim=0)
        pred_target = x_encode[seq_length / 2:].detach()

        loss = self.criterion(pred_target, pred_out)
        return loss

    def _initialize_weights(self):
        # init weights for all submodules
        MobileReg._initialize_weights(self)

        for m in self.modules():
            if isinstance(m, nn.LSTM):
                #print type(m)
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal_(param)

if __name__ == "__main__":  # test
    from utils import get_path
    from dataset import TrackingLabelDataset, FolderUnlabelDataset, DataLoader
    # prepare data
    imgdataset = TrackingLabelDataset("duke-train",
                                      data_file=get_path("DukeMCMT/trainval_duke.txt"), data_aug=True)
    unlabelset = FolderUnlabelDataset("duke-unlabel", data_file="../data/duke_unlabeldata.pkl",
                                      seq_length=24, data_aug=True, extend=True)
    dataloader = DataLoader(imgdataset, batch_size=32)
    unlabelloader = DataLoader(unlabelset)

    model = MobileEncoderReg()
    model.load_mobilenet('pretrained_models/mobilenet_v1_0.50_224.pth')

    for ind in range(1, 5):
        sample = dataloader.next_sample()
        sample_unlabel = unlabelloader.next_sample().squeeze()

        loss = model.forward_combine(sample['img'], sample['label'], sample_unlabel)


        print "iter {}, loss {} {}".format(ind, loss["label"].item(), loss["unlabel"].item())

    print "Finished" # why no auto terminate ?