import math
import random
import torch
import torch.nn as nn

from hdeReg import HDEReg
from extractor import MobileExtractor

Lamb = 0.1 

class MobileReg(HDEReg):

    def __init__(self, hidNum=256, output_type="reg", lamb=0.1, thresh=0.005, device=None):
        # input size should be [192x192]
        self.lamb = lamb
        self.thresh = thresh

        HDEReg.__init__(self, hidNum, output_type, device, init=False)

        self.feature = MobileExtractor(
            hidNum, depth_multiplier=0.5, device=device)    # reinited, could be better

        self._initialize_weights()
        self.load_to_device()

    def load_mobilenet(self, fname):
        self.feature.load_from_npz(fname)

    def unlabel_loss(self, output, threshold):
        """
        :param output: network unlabel output tensor
        :return: unlabel loss tensor
        """
        output = output.to(self.device)
        unlabel_batch = output.shape[0]
        loss = torch.Tensor([0]).to(self.device).float()
        threshold = torch.tensor(threshold).to(self.device).float()

        for ind1 in range(unlabel_batch - 5):  # try to make every sample contribute
            # randomly pick two other samples
            ind2 = random.randint(ind1 + 2, unlabel_batch - 1)  # big distance
            ind3 = random.randint(ind1 + 1, ind2 - 1)  # small distance

            diff_big = torch.sum(
                (output[ind1] - output[ind2]) ** 2).float() / 2.0
            diff_small = torch.sum(
                (output[ind1] - output[ind3]) ** 2).float() / 2.0

            cost = torch.max(diff_small - diff_big - threshold,
                             torch.tensor(0).to(self.device).float())
            loss += cost

        return loss

    def forward_unlabel(self, inputs):
        """
        :param sample: unlabeled data
        :return: unlabel loss
        """
        inputs = inputs.to(self.device)
        outputs = self(inputs)

        loss = self.unlabel_loss(outputs, self.thresh)
        return loss.to(self.device).float()

    def forward_combine(self, inputs, targets, inputs_unlabel):
        loss_label = self.forward_label(inputs, targets)
        loss_unlabel = self.forward_unlabel(inputs_unlabel)
        loss_total = loss_label + self.lamb * loss_unlabel

        loss = {"total": loss_total,
                "label": loss_label,
                "unlabel": loss_unlabel}

        return loss

if __name__ == '__main__':
    import sys
    sys.path.insert(0, "..")
    from utils import get_path, seq_show

    import torch.optim as optim
    from dataset import SingleLabelDataset, DukeSeqLabelDataset, DataLoader

    net = MobileReg()
    net.load_mobilenet('pretrained_models/mobilenet_v1_0.50_224.pth')

    dataset = SingleLabelDataset(
        "duke-test", data_file=get_path('DukeMTMC/test/test.csv'), data_aug=True)
    dataset.shuffle()
    loader = DataLoader(dataset, batch_size=32)
    

    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for ind in range(1, 5000):
        sample = loader.next_sample()
        imgseq = sample[0].squeeze()
        labels = sample[1].squeeze()

        l = net.forward_label(imgseq, labels)
        print l.item()

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        #seq_show(imgseq.numpy(), dir_seq=output.to("cpu").detach().numpy())

    print "Finished"
