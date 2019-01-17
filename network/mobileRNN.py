import torch.nn as nn
import torch.nn.functional as FF

from hdeNet import HDENet
from hdeReg import HDEReg
from extractor import MobileExtractor


class MobileRNN(HDEReg):

    def __init__(self, rnn_type="gru", hidNum=256, rnnHidNum=8, n_layer=2, regNum=2, device=None):
        HDENet.__init__(self, device)

        self.criterion = nn.MSELoss()
        self.feature = MobileExtractor(
            hidNum, depth_multiplier=0.5, device=device)

        if rnn_type == "gru":
            self.rnn = nn.GRU(hidNum, rnnHidNum, n_layer)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(hidNum, rnnHidNum, n_layer)

        self.reg = nn.Linear(rnnHidNum, regNum)

        self._initialize_weights()
        self.load_to_device()

    def forward(self, x):
        seq_length = x.shape[0]

        x = x.to(self.device)
        x = self.feature(x)
        x = x.unsqueeze(1)

        x, h_n = self.rnn(x)
        x = x.view(seq_length, -1)

        output = self.reg(x)
        return output

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from utils import get_path, seq_show
    from dataset import DukeSeqLabelDataset, DataLoader

    import torch.optim as optim

    dataset = DukeSeqLabelDataset(
        "duke-test", data_file=get_path('DukeMTMC/val/person.csv'), seq_length=24)
    dataset.shuffle()
    loader = DataLoader(dataset)

    model = MobileRNN(rnn_type="gru", n_layer=8, rnnHidNum=32)
    model.feature.load_from_npz('pretrained_models/mobilenet_v1_0.50_224.pth')

    optimizer = optim.Adam(model.parameters(), lr=0.03)

    for ind in range(1, 1000):
        sample = loader.next_sample()
        imgseq = sample[0].squeeze()
        labels = sample[1].squeeze()

        loss = model.forward_label(imgseq, labels)
        print loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #seq_show(imgseq.numpy(), dir_seq=output.to("cpu").detach().numpy())

    print "Finished"
