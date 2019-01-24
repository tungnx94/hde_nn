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

    def train(self):
        """ train on one batch """
        self.countTrain += 1
        loss = self.train_loss()

        # backpropagate
        self.optimizer.zero_grad()
        loss[2].backward()
        self.optimizer.step()

    def validate(self):
        """ update val loss history """
        self.logger.info("validation")

        losses = []
        for count in range(self.valStep):
            loss_t = self.train_loss()
            loss_v = self.val_loss()
            losses.append(torch.tensor(tuple(loss_t, loss_v)).unsqueeze(0)) 

        losses = torch.cat(tuple(losses), dim=0)
        loss_mean = torch.mean(losses, dim=0)

        for idx, av in enumerate self.config['losses']
            self.push_to_av(av, loss[idx].item(), self.countTrain)
