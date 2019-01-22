import sys
sys.path.append("..")

from netWF import TestWF
from utils import get_path, seq_show
from dataset import *

LabelSeqLength = 24  # 32
TestStep = 5000 # number of test() calls, 5000
showFreq = 25
Snapshot = 500

Visualize = True

class TestSSWF(TestWF):

    def __init__(self, workingDir, prefix):
        self.visualize = Visualize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        TestWF.__init__(self, workingDir, prefix,
                        testStep=TestStep, saveFreq=Snapshot, showFreq=showFreq)

    def visualize_output(self, inputs, outputs):
        seq_show(inputs.cpu().numpy(), dir_seq=outputs.detach().cpu().numpy(),
                 scale=0.8, mean=self.mean, std=self.std)


class TestLabelSeqWF(TestSSWF):  # Type 1

    def __init__(self, workingDir, prefix):
        self.acvs = {"total": 20,
                     "label": 20,
                     "unlabel": 20}

        TestSSWF.__init__(self, workingDir, prefix)

        self.add_plotter("total_loss", ['total'], [True])
        self.add_plotter("label_loss", ['label'], [True])
        self.add_plotter("unlabel_loss", ['unlabel'], [True])

    def load_dataset(self):
        self.test_dataset = DukeSeqLabelDataset("DUKE-test", data_file=get_path('DukeMTMC/test/test.csv'), seq_length=LabelSeqLength, data_aug=True,
                                   mean=self.mean, std=self.std)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1)

    def test(self):
        sample = self.test_loader.next_sample()
        inputs = sample[0].squeeze()
        targets = sample[1].squeeze()
        loss = self.model.forward_combine(inputs, targets, inputs)

        self.AV['label'].push_back(loss[0].item())
        self.AV['unlabel'].push_back(loss[1].item())
        self.AV['total'].push_back(loss[2].item())


class TestLabelWF(TestSSWF):  # Type 2

    def __init__(self, workingDir, prefix):
        self.acvs = {"label": 20}

        TestSSWF.__init__(self, workingDir, prefix)

        self.add_plotter("label_loss", ['label'], [True])

    def load_dataset(self):
        self.testBatch = 50
        self.test_dataset = SingleLabelDataset("DRONE-test", get_path('DRONE_label'), data_aug=False,
                                  mean=self.mean, std=self.std)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.testBatch)

    def test(self):
        sample = self.test_loader.next_sample()
        loss = self.model.forward_label(sample[0], sample[1])

        self.AV['label'].push_back(loss.item())


class TestUnlabelSeqWF(TestSSWF):  # Type 3

    def __init__(self, workingDir, prefix):
        self.acvs = {"unlabel": 20}

        TestSSWF.__init__(self, workingDir, prefix)

        self.add_plotter("unlabel_loss", ['unlabel'], [True])

    def load_dataset(self):
        self.test_dataset = FolderUnlabelDataset("DRONE-seq", get_path('DRONE_seq'), data_aug=False,
                                    mean=self.mean, std=self.std)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1)

    def test(self):
        sample = self.test_loader.next_sample()
        inputs = sample.squeeze()
        loss = self.model.forward_unlabel(inputs)

        self.AV['unlabel'].push_back(loss.item())
