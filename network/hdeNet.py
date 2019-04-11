import os
import math 
import torch
import torch.nn as nn

class HDENet(torch.nn.Module):

    def __init__(self, device=None):
        torch.nn.Module.__init__(self)
        self.countTrain = 0
        self.device = device

        if device is None:  # select default if not specified
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def new_variable(self, tensor, **kwargs):
        var = torch.autograd.Variable(tensor, **kwargs)  # deprecated
        if self.device == torch.device("cuda"):
            var = var.cuda()
        return var

    def load_to_device(self):
        if self.device == torch.device("cuda"):
            self.cuda()
            print("Using CUDA")

    def load_from_npz(self, file):
        model_dict = self.state_dict()

        preTrainDict = torch.load(file)
        preTrainDict = {k: v for k, v in preTrainDict.items()
                        if k in model_dict}

        model_dict.update(preTrainDict)
        self.load_state_dict(model_dict)

    def load_pretrained(self, file):
        # file needs to point to a relative path
        modelname = os.path.splitext(os.path.basename(file))[0]
        self.countTrain = int(modelname.split('_')[-1])
        self.load_from_npz(file)

        self.load_to_device()

    def _initialize_weights(self):
        # init weights for all submodules
        for m in self.modules():
            # print type(m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal_(param)
