import torch
import torch.nn as nn

from .hdeReg import HDEReg


class MobileRNN(HDEReg):

    def __init__(self, extractor, hidNum=256, rnnHidNum=128, n_layer=4, rnn_type="gru", device=None):
        HDEReg.__init__(self, extractor, hidNum, output_type, device, init=False)

        if rnn_type == "gru":
            self.rnn = nn.GRU(hidNum, rnnHidNum, n_layer)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(hidNum, rnnHidNum, n_layer)

        self.reg = nn.Linear(rnnHidNum, 2)

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
        "duke", path=get_path('DukeMTMC/train.csv'), seq_length=16, data_aug=True)
    dataset.shuffle()
    # dataset.resize(5000)
    loader = DataLoader(dataset)

    model = MobileRNN(rnn_type="gru", n_layer=2, rnnHidNum=128)
    #model.feature.load_from_npz('pretrained_models/mobilenet_v1_0.50_224.pth')

    optimizer = optim.Adam(model.parameters(), lr=0.0075)

    # train
    for ind in range(1, 20):
        sample = loader.next_sample()
        imgseq = sample[0].squeeze()
        labels = sample[1].squeeze()

        # print(imgseq.shape, labels.shape)
        
        #loss = model.forward_label(imgseq, labels)
        loss_w = model.loss_weighted(imgseq, labels, mean=True)
        loss = model.loss_label(imgseq, labels, mean=True)
        print(loss_w.item() , ' ', loss.item())

        optimizer.zero_grad()
        loss_w.backward()
        optimizer.step()

    print("Finished")
