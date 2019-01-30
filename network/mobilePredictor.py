# based on MobileReg and StateEncoderDecoder
import torch
import torch.nn as nn

from hdeNet import HDENet
from hdeReg import HDEReg
from mobileReg import MobileReg
from extractor import MobileExtractor


class MobileEncoderReg(MobileReg):

    def __init__(self, hidNum=256, rnnHidNum=128, output_type="reg", lamb=0.01, device=None):
        # input tensor should be [Batch, 3, 192, 192]
        self.lamb = lamb
        self.hidNum = hidNum
        self.rnnHidNum = rnnHidNum

        HDEReg.__init__(self, hidNum, output_type, device, init=False)

        self.feature = MobileExtractor(hidNum, depth_multiplier=0.5, device=device) 
        # self.reg = nn.Linear(hidNum, regNum)  # regression (sine, cosine)

        self.pred_en = nn.GRU(hidNum, rnnHidNum)
        self.pred_de = nn.GRU(hidNum, rnnHidNum)
        self.pred_de_linear = nn.Linear(rnnHidNum, hidNum)  # FC

        self.load_to_device()
        self._initialize_weights()

    def loss_unlabel(self, inputs, mean=True):
        # input size is [SeqLength, 3, W, H]
        inputs = inputs.to(self.device)

        x_encode = self.feature(inputs)
        seq_length = x_encode.size()[0]  # SeqLength

        # prediction: use first half as input, last half as target
        innum = seq_length / 2

        # input of LSTM is [SeqLength x Batch x InputSize]
        # output = [SeqLength x Batch x HiddenSize], (hidden_n, cell_n)
        pred_in = x_encode[:innum].unsqueeze(1)  # add batch=1 dimension
        hidden_0 = self.new_variable(torch.zeros(
            1, 1, self.rnnHidNum))

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
        
        loss = self.criterion(pred_out, pred_target)
        if mean:
            loss = torch.mean(loss)
        return loss


if __name__ == "__main__":  # test
    import sys
    sys.path.insert(0, "..")
    from utils import get_path
    from dataset import SingleLabelDataset, FolderUnlabelDataset, DataLoader

    # prepare data
    imgdataset = SingleLabelDataset("duke-train",
                                    path=get_path("DukeMTMC/train/train.csv"))
    unlabelset = FolderUnlabelDataset(
        "duke-unlabel", path=get_path("DukeMTMC/train/images_unlabel"))

    dataloader = DataLoader(imgdataset, batch_size=32)
    unlabelloader = DataLoader(unlabelset)

    model = MobileEncoderReg()
    model.load_mobilenet('pretrained_models/mobilenet_v1_0.50_224.pth')

    for ind in range(1, 10):
        sample = dataloader.next_sample()
        sample_unlabel = unlabelloader.next_sample().squeeze()

        loss = model.loss_combine(
            sample[0], sample[1], sample_unlabel, mean=True)

        print "iter {}, loss {} {}".format(ind, loss[0].item(), loss[1].item())

    print "Finished"  # no auto terminate ?
