import torch
import torch.nn as nn

from .hdeReg import HDEReg

class HDE_RNN(HDEReg):

    def __init__(self, extractor, hidNum=256, rnnHidNum=1024, device=None):
        print("SIMPLE RNN")
        self.hidNum = hidNum
        self.rnnHidNum = rnnHidNum

        HDEReg.__init__(self, extractor, hidNum, device, init=False)

        self.i2h = nn.Sequential(
            nn.Linear(hidNum + rnnHidNum, rnnHidNum)
            #nn.ReLU()
        )
        self.i2o = nn.Sequential(
            nn.Linear(hidNum + rnnHidNum, hidNum)
            #nn.ReLU()
        )
        self.reg = nn.Sequential(
            nn.Linear(hidNum, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self._initialize_weights()
        self.load_to_device()

    def forward_one_step(self, x, hidden):
        combined = torch.cat((x, hidden))
        hidden = self.i2h(combined)
        x = self.i2o(combined)
        x = self.reg(x)

        return x, hidden

    def forward(self, x):
        seq_length = x.shape[0]
        #print(seq_length)
        x = x.to(self.device)
        x = self.feature(x).squeeze()

        hidden = torch.zeros(self.rnnHidNum).to(self.device)
        outputs = []

        for i in range(seq_length):
            out, hidden = self.forward_one_step(x[i], hidden)
            outputs.append(out)

        return torch.stack(outputs)