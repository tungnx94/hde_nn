# Vanilla model CNN
import torch.nn as nn

from hdeNet import HDENet
from extractor import BaseExtractor


class HDEReg(HDENet):

    def __init__(self, hidNum=256, device=None):
        # input size should be [192x192]
        HDENet.__init__(self, device)
        self.criterion = nn.MSELoss()  # L2

        self.feature = BaseExtractor()
        self.reg = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),

            nn.Linear(64, 16),
            nn.Tanh(),

            nn.Linear(16, 2),
        )

        self._initialize_weights()
        self.load_to_device()

    def forward(self, x):
        x = x.to(self.device)
        x = self.feature(x).squeeze()
        x = self.reg(x)
        return x

    def forward_label(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self(inputs)
        return self.criterion(outputs, targets)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, "..")

    import torch.optim as optim
    from utils import get_path
    from dataset import SingleLabelDataset, DataLoader

    net = HDEReg()
    dataset = SingleLabelDataset(
        "duke", data_file=get_path('DukeMTMC/val/person.csv'), img_size=96)
    dataset.shuffle()
    loader = DataLoader(dataset, batch_size=32)

    optimizer = optim.Adam(net.parameters(), lr=0.03)
    for ind in range(1, 500):
        sample = loader.next_sample()
        imgseq = sample[0].squeeze()
        labels = sample[1].squeeze()

        loss = net.forward_label(imgseq, labels)
        print loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print "Finished"
