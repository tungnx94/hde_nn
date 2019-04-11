import torch
import torch.nn as nn

from .hdeReg import HDEReg

# TODO: fix forward()
class MobileRNN(HDEReg):

    def __init__(self, config, device=None):
        HDEReg.__init__(self, config, device, init=False)

        self.rnnHidNum = self.config["rnn_hid_num"]
        self.rnn_layers = self.params["layers"]
        self.rnn_cell = self.params["cell"]

        if self.rnn_cell == "gru":
            self.rnn = nn.GRU(self.hidNum, self.rnnHidNum, self.rnn_layers)
        elif self.rnn_cell == "lstm":
            self.rnn = nn.LSTM(self.hidNum, self.rnnHidNum, self.n_layer)

        self.reg = nn.Sequential(
            nn.Linear(self.rnnHidNum, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self._initialize_weights()
        self.load_to_device()

    # need to fix this to allow batch size > 1 
    def forward(self, x):
        batch = x.shape[0]
        seq_length = x.shape[1]

        x = x.to(self.device)
        x = self.feature(x).squeeze()
        
        x = x.unsqueeze(1)
        x, h_n = self.rnn(x)
        x = x.view(seq_length, -1)
        output = self.reg(x)
        return output