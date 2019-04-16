import torch
import torch.nn as nn

from .hdeReg import HDEReg

class FeatureGRU(HDEReg):

    def __init__(self, config, device=None):
        HDEReg.__init__(self, config, device, init=False)

        self.rnnHidNum = self.params["rnn_hid_num"]
        self.rnn_layers = self.params["layers"]
        self.fNum = self.params["f_num"]

        self.rnn = nn.GRU(self.fNum, self.rnnHidNum, self.rnn_layers)
        self.reg = nn.Linear(self.rnnHidNum, 2)

        """
        self.reg = nn.Sequential(
            nn.Linear(self.rnnHidNum, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )"""

        self._initialize_weights()
        self.load_to_device()

    def forward(self, x):
        batch = x.shape[0]
        x = x.to(self.device)

        x = self.feature(x).view(batch, -1)
        seq_length = x.shape[1] // self.fNum
        x = x.view(batch, seq_length, -1)
        # x = [batch x seq_length x self.fNum]
        x = x.permute(1, 0, 2)

        # x = [seq_length x batch x self.fNum] 
        x, _ = self.rnn(x)
        # x = [seq_length x batch x rnnHidNum]

        y = self.reg(x[-1])
        return y