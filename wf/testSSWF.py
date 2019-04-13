

from .testWF import TestWF
import numpy as np

# TODO: need to complete
class TestLabelWF(TestWF):  # Type 1

    def val_metrics(self, sample):
        outputs = self.model(sample[0])
        loss = self.model.loss_label(sample[0], sample[1], mean=True).item()

        angle_err, angle_acc = utils.angle_metric(outputs, sample[1])
        values = [loss, angle_loss, angle_acc]

        return np.array(values)


class TestLabelSeqWF(TestWF):  # Type 2

    def val_metrics(self, sample):
        # test a batch of size possible > 1
        losses = []
        for sample_seq in sample:
            inputs, targets = sample_seq
            outputs = self.model(inputs)

            loss = self.model.loss_combine(inputs, targets, inputs, mean=True)
            angle_err, angle_acc = utils.angle_metric(outputs, targets)

            values = [loss[0].item(), loss[1].item(), loss[2].item(), angle_err, angle_acc]
            losses.append(np.array(values))

        losses = np.mean(np.array(losses), axis=0)
        return losses

class TestUnlabelWF(TestWF):  # Type 3

    def val_metrics(self, sample):
        losses = []
        for sample_seq in sample:
            loss = self.model.forward_unlabel(sample_seq).item()
            losses.append(loss)

        losses = np.mean(np.array(losses), axis=0)
        return losses
