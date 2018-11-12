import torch
import config as cnf

from utils import unlabel_loss, angle_metric
from folderLabelData import FolderLabelDataset
from folderUnlabelData import FolderUnlabelDataset
from dukeSeqLabelData import DukeSeqLabelDataset

UnlabelBatch = 24  # 32
TestStep = 10000  # number of test() calls

test_label_file = '/datadrive/person/DukeMTMC/test_heading_gt.txt'
test_label_img_folder = '/home/wenshan/headingdata/val_drone'
test_unlabel_img_folder = '/datadrive/exp_bags/20180811_gascola'

class TestWF(GeneralWF):

    def run():
        for iteration in range(TestStep):
            self.test()
        print "Finished testing"


class TestLabelSeqWF(TestWF):  # Type 1

    def get_test_dataset()
        return DukeSeqLabelDataset(labelfile=test_label_file, batch=UnlabelBatch, data_aug=True,
                                   mean=self.mean, std=self.std)


class TestFolderWF(TestWF):  # Type 2

    def get_test_dataset():
        self.testBatch = 50
        return FolderLabelDataset(img_dir=test_label_img_folder, data_aug=False,
                                  mean=self.mean, std=self.std)

    def calculate_loss(self, val_sample):
        """ label loss only """
        inputImgs = val_sample['img'].to(self.device)
        labels = val_sample['label'].to(self.device)

        output = self.model(inputImgs)
        loss_label = self.criterion(output, labels).item()

        if visualize:
            self.visualize_output(inputImgs, output)
            angle_error, cls_accuracy = angle_metric(
                output.detach().cpu().numpy(), labels.cpu().numpy())
            print 'label-loss %.4f, angle diff %.4f, accuracy %.4f' % (loss_label, angle_error, cls_accuracy)

        return loss_label


class TestUnlabelSeqWF(TestWF):  # Type 3

    def get_test_dataset():
        return FolderUnlabelDataset(img_dir=test_unlabel_img_folder, data_aug=False, include_all=True,
                                    mean=self.mean, std=self.std)

    def calculate_loss(self, val_sample):
        """ unlabel loss only """
        inputImgs = val_sample.squeeze().to(self.device)
        output = self.model(inputImgs)

        loss_unlabel = unlabel_loss(output.numpy(), Thresh)

        # import ipdb;ipdb.set_trace()
        if visualize:
            self.visualize_output(inputImgs, output)
            print loss_unlabel

        return torch.tensor([loss_unlabel])
