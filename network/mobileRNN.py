import torch
import torch.nn as nn
import torch.nn.functional as FF

from hdeNet import HDENet
from hdeReg import HDEReg
from extractor import MobileExtractor, BaseExtractor


class MobileRNN(HDEReg):

    def __init__(self, rnn_type="gru", hidNum=256, rnnHidNum=64, n_layer=2, regNum=2, device=None):
        HDENet.__init__(self, device)

        self.criterion = nn.MSELoss()
        self.feature = MobileExtractor(
            hidNum, depth_multiplier=0.5, device=device)

        self.n_layer = n_layer
        self.hidNum = hidNum
        self.rnnHidNum = rnnHidNum

        #self.feature = BaseExtractor()

        if rnn_type == "gru":
            self.rnn = nn.GRU(hidNum, rnnHidNum, n_layer)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(hidNum, rnnHidNum, n_layer)

        self.reg = nn.Linear(rnnHidNum, regNum)

        """
        self.reg = nn.Sequential(
            nn.Linear(rnnHidNum, rnnHidNum/2),
            nn.ReLU(),

            nn.Linear(rnnHidNum/2, regNum)
        )
        """

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

        """ # sequential proceeding
        h = torch.zeros(self.n_layer, 1, self.rnnHidNum) 
        h = self.new_variable(h)
        out_seq = []
        for t in range(seq_length):
            y = x[t].view(1,1,-1)
            o, h = self.rnn(y, h)

            out_t = self.reg(o.view(1, self.rnnHidNum))
            out_seq.append(out_t) 

        pred_out = torch.cat(tuple(out_seq), dim=0)
        return pred_out
        """

    def loss_weighted(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self(inputs)
        
        seq_length = outputs.shape[0]
        diff = torch.norm(targets-outputs, dim=1)
        
        weight = [0.1]
        for t in range(seq_length-1):
            w = torch.norm(targets[t+1]-targets[t]).item()
            weight.append(w)

        weight = torch.tensor(weight).to(self.device)
        weight = torch.exp(weight)
        loss = weight * diff
        return torch.mean(loss)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils import get_path, seq_show
    from dataset import DukeSeqLabelDataset, DataLoader

    import torch.optim as optim

    dataset = DukeSeqLabelDataset(
        "duke-test", data_file=get_path('DukeMTMC/val/person.csv'), seq_length=8, data_aug=True)
    dataset.shuffle()
    dataset.resize(5000)
    loader = DataLoader(dataset)

    model = MobileRNN(rnn_type="gru", n_layer=2, rnnHidNum=128)
    #model.feature.load_from_npz('pretrained_models/mobilenet_v1_0.50_224.pth')

    optimizer = optim.Adam(model.parameters(), lr=0.0075)

    # train
    for ind in range(1, 10000):
        sample = loader.next_sample()
        imgseq = sample[0].squeeze()
        labels = sample[1].squeeze()

        #loss = model.forward_label(imgseq, labels)
        loss_w = model.loss_weighted(imgseq, labels)
        loss = model.forward_label(imgseq, labels)
        print loss_w.item() , ' ', loss.item()

        optimizer.zero_grad()
        loss_w.backward()
        optimizer.step()

    # test

        #seq_show(imgseq.numpy(), dir_seq=output.to("cpu").detach().numpy())

    print "Finished"
