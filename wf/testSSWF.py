import sys
sys.path.append("..")

from netWF import TestWF
from utils import get_path, seq_show

LabelSeqLength = 24  # 32
TestStep = 1000 # number of test() calls, 5000
ShowFreq = 25
Snapshot = 500

Visualize = True # False

class TestSSWF(TestWF):
    # should keep this class ?
    def __init__(self, workingDir, prefix, model_type, trained_model):
        self.visualize = Visualize
        TestWF.__init__(self, workingDir, prefix, model_type, trained_model,
                        testStep=TestStep, saveFreq=Snapshot, showFreq=ShowFreq)

    def visualize_output(self, inputs, outputs):
        seq_show(inputs.cpu().numpy(), dir_seq=outputs.detach().cpu().numpy(),
                 scale=0.8, mean=self.mean, std=self.std)


class TestLabelSeqWF(TestSSWF):  # Type 1

    def __init__(self, workingDir, prefix, model_type, trained_model):
        self.acvs = {"total": 20,
                     "label": 20,
                     "unlabel": 20}

        TestSSWF.__init__(self, workingDir, prefix, model_type, trained_model)

        self.add_plotter("total_loss", ['total'], [True])
        self.add_plotter("label_loss", ['label'], [True])
        self.add_plotter("unlabel_loss", ['unlabel'], [True])

    def test(self):
        sample = self.test_loader.next_sample()
        inputs = sample[0].squeeze()
        targets = sample[1].squeeze()
        loss = self.model.forward_combine(inputs, targets, inputs)

        self.AV['label'].push_back(loss[0].item())
        self.AV['unlabel'].push_back(loss[1].item())
        self.AV['total'].push_back(loss[2].item())


class TestLabelWF(TestSSWF):  # Type 2

    def __init__(self, workingDir, prefix, model_type, trained_model):
        self.acvs = {"label": 20}

        TestSSWF.__init__(self, workingDir, prefix, model_type, trained_model)

        self.add_plotter("label_loss", ['label'], [True])

    def test(self):
        sample = self.test_loader.next_sample()
        loss = self.model.forward_label(sample[0], sample[1])

        self.AV['label'].push_back(loss.item())


class TestUnlabelWF(TestSSWF):  # Type 3

    def __init__(self, workingDir, prefix, model_type, trained_model):
        self.acvs = {"unlabel": 20}

        TestSSWF.__init__(self, workingDir, prefix, model_type, trained_model)

        self.add_plotter("unlabel_loss", ['unlabel'], [True])

    def test(self):
        sample = self.test_loader.next_sample()
        inputs = sample.squeeze()
        loss = self.model.forward_unlabel(inputs)

        self.AV['unlabel'].push_back(loss.item())
