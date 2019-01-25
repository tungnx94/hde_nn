from testWF import TestWF

class TestLabelSeqWF(TestWF):  # Type 1

    def test(self):
        sample = self.test_loader.next_sample()
        inputs = sample[0].squeeze()
        targets = sample[1].squeeze()
        loss = self.model.forward_combine(inputs, targets, inputs)

        self.AV['label'].push_back(loss[0].item())
        self.AV['unlabel'].push_back(loss[1].item())
        self.AV['total'].push_back(loss[2].item())


class TestLabelWF(TestWF):  # Type 2

    def test(self):
        sample = self.test_loader.next_sample()
        loss = self.model.forward_label(sample[0], sample[1])

        self.AV['label'].push_back(loss.item())


class TestUnlabelWF(TestWF):  # Type 3

    def test(self):
        sample = self.test_loader.next_sample()
        inputs = sample.squeeze()
        loss = self.model.forward_unlabel(inputs)

        self.AV['unlabel'].push_back(loss.item())
