from .testWF import TestWF

# TODO: need to be run
class TestLabelWF(TestWF):  # Type 1

    def test(self, sample):
        loss = self.model.forward_label(sample[0], sample[1], mean=True)

        self.push_to_av("label", loss.item(), self.iteration)

class TestLabelSeqWF(TestWF):  # Type 2

    def test(self, sample):
        inputs = sample[0].squeeze()
        targets = sample[1].squeeze()
        loss = self.model.forward_combine(inputs, targets, inputs, mean=True)

        self.push_to_av("label", loss[0].item(), self.iteration)
        self.push_to_av("unlabel", loss[1].item(), self.iteration)
        self.push_to_av("total", loss[2].item(), self.iteration)

class TestUnlabelWF(TestWF):  # Type 3

    def test(self, sample):
        inputs = sample.squeeze()
        loss = self.model.forward_unlabel(inputs)

        self.push_to_av("unlabel", loss.item(), self.iteration)
