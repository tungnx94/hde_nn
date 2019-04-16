import torch
import torch.nn as nn

from .hdeReg import HDEReg

class HDE_RNN(HDEReg):

    def __init__(self, config, device=None):
        HDEReg.__init__(self, config, device, init=False)

        self.rnnHidNum = self.params["rnn_hid_num"]

        self.i2h = nn.Sequential(
            nn.Linear(self.hidNum + self.rnnHidNum, self.rnnHidNum),
            nn.ReLU()
        )
        self.i2o = nn.Sequential(
            nn.Linear(self.hidNum + self.rnnHidNum, self.hidNum),
            nn.ReLU()
        )
        # self.reg initilized in HDEReg

        self._initialize_weights()
        self.load_to_device()

    def forward_one_step(self, x, hidden):
        combined = torch.cat((x, hidden))
        hidden = self.i2h(combined)
        x = self.i2o(combined)
        x = self.reg(x)

        return x, hidden

    def forward(self, x):
        squeezed = len(x.shape) == 4
        if squeezed:
            x = x.unsqueeze(1)

        batch = x.shape[0]
        seq_length = x.shape[1]
        x = x.to(self.device)
        # x = [batch x seq_length x 192x192x3]
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
        if squeezed:
            z = z.squeeze(dim=1)
        return z