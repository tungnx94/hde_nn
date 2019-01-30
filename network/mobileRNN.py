import torch
import torch.nn as nn
import torch.nn.functional as FF

from hdeReg import HDEReg
from extractor import MobileExtractor, BaseExtractor


class MobileRNN(HDEReg):

    def __init__(self, hidNum=256, rnnHidNum=128, n_layer=2, rnn_type="gru", output_type="reg", device=None):
        HDEReg.__init__(self, hidNum, output_type, device, init=False)

        # self.feature = BaseExtractor()
        self.feature = MobileExtractor(
            hidNum, depth_multiplier=0.5, device=device)

        if rnn_type == "gru":
            self.rnn = nn.GRU(hidNum, rnnHidNum, n_layer)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(hidNum, rnnHidNum, n_layer)

        if output_type == "reg": # regressor
            self.reg = nn.Linear(rnnHidNum, 2)
        else:
            self.reg = nn.Linear(rnnHidNum, 8)

        self._initialize_weights()
        self.load_to_device()

    def forward(self, x):
        seq_length = x.shape[0]
        x = x.to(self.device)
        x = self.feature(x).squeeze()
        
        x = x.unsqueeze(1)
        x, h_n = self.rnn(x)
        x = x.view(seq_length, -1)
        output = self.reg(x)
        return output

    def loss_weighted(self, inputs, targets, mean=False):
        seq_length = inputs.shape[0]

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        loss = self.loss_label(inputs, targets)
        
        weight = [0.1]
        for t in range(seq_length-1):
            w = torch.norm(targets[t+1]-targets[t]).item()
            weight.append(w)

        weight = torch.tensor(weight).to(self.device)
        weight = torch.exp(weight)

        loss = torch.mean(loss, dim=1) * weight

        if mean:
            loss = torch.mean(loss)
        return loss

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils import get_path, seq_show
    from dataset import DukeSeqLabelDataset, DataLoader

    import torch.optim as optim

    dataset = DukeSeqLabelDataset(
        "duke", path=get_path('DukeMTMC/train/train.csv'), seq_length=8, data_aug=True)
    dataset.shuffle()
    # dataset.resize(5000)
    loader = DataLoader(dataset)

    model = MobileRNN(rnn_type="gru", n_layer=2, rnnHidNum=128)
    #model.feature.load_from_npz('pretrained_models/mobilenet_v1_0.50_224.pth')

    optimizer = optim.Adam(model.parameters(), lr=0.0075)

    # train
    for ind in range(1, 100):
        sample = loader.next_sample()
        imgseq = sample[0].squeeze()
        labels = sample[1].squeeze()

        print imgseq.shape
        print labels.shape
        
        #loss = model.forward_label(imgseq, labels)
        loss_w = model.loss_weighted(imgseq, labels, mean=True)
        loss = model.loss_label(imgseq, labels, mean=True)
        print loss_w.item() , ' ', loss.item()

        optimizer.zero_grad()
        loss_w.backward()
        optimizer.step()

    # test

        #seq_show(imgseq.numpy(), dir_seq=output.to("cpu").detach().numpy())

    print "Finished"
