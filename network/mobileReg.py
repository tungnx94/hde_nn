import math
import random
import torch
import torch.nn as nn

from .hdeReg import HDEReg

Lamb = 0.1 

class MobileReg(HDEReg):

    def __init__(self, config, device=None):
        # input size should be [192x192]
        HDEReg.__init__(self, config, device, init=False)

        self.lamb = self.params["lamb"]
        self.thresh = self.params["thresh"]

        self.load_to_device()
        self._initialize_weights()

    def loss_unlabel(self, inputs):
        """
        :param output: network unlabel output tensor
        :return: unlabel loss tensor
        """
        inputs = inputs.to(self.device)
        outputs = self(inputs)

        unlabel_batch = outputs.shape[0]
        loss = torch.Tensor([0]).to(self.device).float()
        threshold = torch.tensor(self.thresh).to(self.device).float()

        for ind1 in range(unlabel_batch - 5):  # try to make every sample contribute
            # randomly pick two other samples
            ind2 = random.randint(ind1 + 2, unlabel_batch - 1)  # big distance
            ind3 = random.randint(ind1 + 1, ind2 - 1)  # small distance

            diff_big = torch.sum(
                (outputs[ind1] - outputs[ind2]) ** 2).float() / 2.0
            diff_small = torch.sum(
                (outputs[ind1] - outputs[ind3]) ** 2).float() / 2.0

            cost = torch.max(diff_small - diff_big - threshold,
                             torch.tensor(0).to(self.device).float())
            loss += cost

        loss = loss.to(self.device).float()
        return loss

    def loss_combine(self, inputs, targets, inputs_unlabel, mean=False):
        loss_label = self.loss_label(inputs, targets, mean)
        loss_unlabel = self.loss_unlabel(inputs_unlabel)
        loss_total = torch.mean(loss_label) + self.lamb * loss_unlabel / inputs.shape[0] # should be divided by seq_length or not ?

        return (loss_label, loss_unlabel, loss_total)
