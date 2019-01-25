import torch
import torch.optim as optim

from trainWF import TrainWF

class TrainSSWF(TrainWF):

    def __init__(self, config):
        TrainWF.__init__(self, config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def prepare_dataset(self, dloader):
        label_dts, unlabel_dts, val_dts = self.load_dataset()

        self.train_loader = dloader.loader(label_dts, self.batch)
        self.train_unlabel_loader = dloader.loader(unlabel_dts, self.batch_unlabel)
        self.val_loader = dloader.loader(val_dts, self.batch_val)

    def train_loss(self):
        # get next samples
        inputs, targets, _ = self.train_loader.next_sample()
        unlabel_seqs = self.train_unlabel_loader.next_sample().squeeze()  # remove 0-dim (=1)

        return self.model.forward_combine(inputs, targets, unlabel_seqs)

    def backward_loss(self):
        # return combined loss
        loss = self.train_loss()
        return loss[2] 
