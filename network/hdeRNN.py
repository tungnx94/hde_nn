import torch
import torch.nn as nn

from .hdeReg import HDEReg

class HDE_RNN(HDEReg):

    def __init__(self, extractor, hidNum=256, rnnHidNum=128, device=None):
        self.hidNum = hidNum
        self.rnnHidNum = rnnHidNum

        HDEReg.__init__(self, extractor, hidNum, device, init=False)

        self.i2h = nn.Sequential(
            nn.Linear(hidNum + rnnHidNum, rnnHidNum),
            nn.ReLU()
        )
        self.i2o = nn.Sequential(
            nn.Linear(hidNum + rnnHidNum, hidNum),
            nn.ReLU()
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
        batch = x.shape[0]
        seq_length = x.shape[1]
        x = x.to(self.device)

        """
        x = self.feature(x).squeeze()
        
        hidden = torch.zeros(self.rnnHidNum).to(self.device)
        outputs = []

        for i in range(seq_length):
            out, hidden = self.forward_one_step(x[i], hidden)
            outputs.append(out)

        outputs = torch.stack(outputs)
        return outputs
        """


        z = []
        for i in range(batch):
            y = self.feature(x[i]).view(seq_length, -1)

            hidden = torch.zeros(self.rnnHidNum).to(self.device)
            outputs = []
            for j in range(seq_length):
                out, hidden = self.forward_one_step(y[j], hidden)
                outputs.append(out)

            outputs = torch.stack(outputs)
            z.append(outputs)

        z = torch.stack(z)
        return z