import torch
import torch.nn as nn

from .hdeReg import HDEReg

class FeatureRNN(HDEReg):

    def __init__(self, config, device=None):
        HDEReg.__init__(self, config, device, init=False)

        self.rnnHidNum = self.params["rnn_hid_num"]
        self.fNum = self.params["f_num"]

        self.i2h = nn.Sequential(
            nn.Linear(self.fNum + self.rnnHidNum, self.rnnHidNum),
            nn.ReLU()
        )
        self.i2o = nn.Sequential(
            nn.Linear(self.fNum + self.rnnHidNum, self.fNum),
            nn.ReLU()
        )
        
        self.reg = nn.Linear(self.fNum, 2)
        """
        self.reg = nn.Sequential(
            nn.Linear(self.fNum, self.fNum / 2),
            nn.ReLU(),
            nn.Linear(self.fNum / 2, 2)
        )
        """

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
        x = x.to(self.device)

        x = self.feature(x).view(batch, -1)

        seq_length = x.shape[1] // self.fNum        
        x = x.view(batch, seq_length, -1)

        z = []
        for batch_sample in x:
            hidden = torch.zeros(self.rnnHidNum).to(self.device)

            for feature_part in batch_sample:
                out, hidden = self.forward_one_step(feature_part, hidden)
            z.append(out)

        z = torch.stack(z)
        return z
        