import torch
import sys
import random

import torch.nn as nn
import torch.optim as optim
import numpy as np

from math import pi
from os.path import join
from torch.utils.data import DataLoader

from workflow import WorkFlow
from utils import loadPretrain2, loadPretrain, seq_show_with_arrow
from MobileReg import MobileReg

from labelData import LabelDataset
from unlabelData import UnlabelDataset
from folderLabelData import FolderLabelDataset
from folderUnlabelData import FolderUnlabelDataset
from dukeSeqLabelData import DukeSeqLabelDataset

sys.path.append('../WorkFlow')


exp_prefix = 'vis_1_3_'  # ?
Batch = 128
UnlabelBatch = 24  # 32
learning_rate = 0.0005  # learning rate
Trainstep = 20000  # number of train() calls
Lamb = 0.1  # ?
Thresh = 0.005  # threshold ?
TestBatch = 1

Snapshot = 5000  # do a snapshot every Snapshot steps (save period)
TestIter = 10  # do a testing every TestIter steps
ShowIter = 1  # print to screen

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# hardcode in labelData, used where ?
train_label_file = '/datadrive/person/DukeMTMC/trainval_duke.txt'
test_label_file = '/datadrive/person/DukeMTMC/test_heading_gt.txt'
unlabel_file = 'duke_unlabeldata.pkl'
saveModelName = 'facing'

test_label_img_folder = '/home/wenshan/headingdata/val_drone'
test_unlabel_img_folder = '/datadrive/exp_bags/20180811_gascola'

pre_mobile_model = 'pretrained_models/mobilenet_v1_0.50_224.pth'
load_pre_mobile = False

pre_model = 'models/1_2_facing_20000.pkl'
load_pre_train = True

TestType = 2  # 0: none, 1: labeled sequence, 2: labeled folder, 3: unlabeled sequence

LogParamList = ['Batch', 'UnlabelBatch', 'learning_rate', 'Trainstep',
                'Lamb', 'Thresh']  # these params will be log into the file


class MyWF(WorkFlow.WorkFlow):

    def __init__(self, workingDir, prefix="", suffix=""):
        super(MyWF, self).__init__(workingDir, prefix, suffix)

        # Record useful params in logfile
        logstr = ''
        for param in LogParamList:
            logstr += param + ': ' + str(globals()[param]) + ', '
        self.logger.info(logstr)

        self.labelEpoch = 0
        self.unlabelEpoch = 0
        self.testEpoch = 0

        self.countTrain = 0
        self.device = 'cuda'
        global TestBatch

        # Data & Dataloaders
        # 1 labeled & 1 unlabeled dataset
        label_dataset = LabelDataset(balence=True, mean=mean, std=std)
        self.train_loader = DataLoader(
            label_dataset, batch_size=Batch, shuffle=True, num_workers=6)

        unlabel_dataset = UnlabelDataset(
            batch=UnlabelBatch, balence=True, mean=mean, std=std)
        self.train_unlabel_loader = DataLoader(
            unlabel_dataset, batch_size=1, shuffle=True, num_workers=4)

        # Test data
        if TestType == 1 or TestType == 0:  # labeled sequence
            testdataset = DukeSeqLabelDataset(
                labelfile=test_label_file, batch=UnlabelBatch, data_aug=True, mean=mean, std=std)
            TestBatch = 1

        elif TestType == 2:  # labeled folder
            testdataset = FolderLabelDataset(
                imgdir=test_label_img_folder, data_aug=False, mean=mean, std=std)
            TestBatch = 50

        elif TestType == 3:  # unlabeled sequence
            testdataset = FolderUnlabelDataset(
                imgdir=test_unlabel_img_folder, data_aug=False, include_all=True, mean=mean, std=std)
            TestBatch = 1

        # Test loader
        self.test_loader = torch.utils.data.DataLoader(
            testdataset, batch_size=TestBatch, shuffle=True, num_workers=1)

        # Data iterators
        self.train_data_iter = iter(self.train_loader)
        self.train_unlabel_iter = iter(self.train_unlabel_loader)
        self.test_data_iter = iter(self.test_loader)

        # Model
        self.model = MobileReg()
        if load_pre_mobile:
            self.model.load_pretrained_pth(pre_mobile_model)

        if load_pre_train:
            loadPretrain(self.model, pre_model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learn)
        self.criterion = nn.MSELoss()

        # self.AV ?
        # self.AVP ?
        self.AV['loss'].avgWidth = 100  # there's a default plotter for 'loss'

        # second param is the number of average data
        self.add_accumulated_value('label_loss', 100)
        self.add_accumulated_value('unlabel_loss', 100)
        self.add_accumulated_value('test_loss', 10)
        self.add_accumulated_value('test_label', 10)
        self.add_accumulated_value('test_unlabel', 10)

        self.AVP.append(WorkFlow.VisdomLinePlotter(
            "total_loss", self.AV, ['loss', 'test_loss'], [True, True]))
        self.AVP.append(WorkFlow.VisdomLinePlotter(
            "label_loss", self.AV, ['label_loss', 'test_label'], [True, True]))
        self.AVP.append(WorkFlow.VisdomLinePlotter("unlabel_loss", self.AV, [
                        'unlabel_loss', 'test_unlabel'], [True, True]))

    def initialize(self, device):
        """ Initilize WF with device """
        super(MyWF, self).initialize()

        self.logger.info("Initialized.")
        self.device = device
        self.model.to(device)

    def finalize(self):
        """ save model and values after training """
        super(MyWF, self).finalize()
        self.print_delimeter('finalize ...')
        self.save_snapshot()

    def load_model(self, model, name):
        """ load the trained parameters from a pickle file :param key """
        # load pretrained dict
        preTrainDict = torch.load(name)
        model_dict = model.state_dict()

        preTrainDict = {k: v for k, v in preTrainDict.items()
                        if k in model_dict}
        # debug
        for item in preTrainDict:
            print('  Load pretrained layer: ', item)

        # update state dict
        model_dict.update(preTrainDict)
        model.load_state_dict(model_dict)
        return model

    def save_model(self, model, name):
        """ Save :param: model to pickle file """
        modelname = self.prefix + name + self.suffix + '.pkl'
        torch.save(model.state_dict(), self.modeldir + '/' + modelname)

    def save_snapshot(self):
        """ write accumulated values and save temporal model """
        self.write_accumulated_values()
        self.draw_accumulated_values()
        self.save_model(self.model, saveModelName + '_' + str(self.countTrain))

    def next_sample(self, data_iter, loader, epoch):
        """ get next batch, update data_iter and epoch if needed """
        try:
            sample = data_iter.next()
        except:
            data_iter = iter(loader)
            sample = data_iter.next()
            epoch += 1

        return sample, data_iter, epoch

    def angle_diff(self, outputs, labels):
        """ compute angular difference """

        # calculate angle from coordiate (x, y)
        output_angle = np.arctan2(outputs[:, 0], outputs[:, 1])
        label_angle = np.arctan2(labels[:, 0], labels[:, 1])

        diff_angle = output_angle - label_angle

        # map difference to (-pi, pi)
        mask = diff_angle < -pi
        diff_angle[mask] = diff_angle[mask] + 2 * pi
        mask = diff_angle > pi
        diff_angle[mask] = diff_angle[mask] - 2 * pi

        # debug
        print output_angle
        print label_angle
        print diff_angle

        return diff_angle

    def angle_loss(self, outputs, labels):
        """ compute mean angular difference between outputs & labels"""
        diff_angle = self.angle_diff(outputs, labels)
        return np.mean(np.abs(diff_angle))

    def accuracy_cls(self, outputs, labels):
        """ 
        compute accuracy 
        :param outputs, labels: numpy array
        """
        diff_angle = self.angle_diff(outputs, labels)
        acc_angle = diff_angle < 0.3927  # 22.5 * pi / 180 = pi/8

        acc = float(np.sum(acc_angle)) / labels.shape[0]
        return acc

    def angle_metric(self, outputs, labels):
        """ return angle loss and accuracy"""
        return self.angle_loss(outputs, labels), self.angle_cls(outputs, labels)

    def unlabel_loss(self, output):
        """
        :param output: network unlabel output
        :return: unlabel loss
        """
        loss_unlabel = torch.Tensor([0]).to(self.device)  # empty tensor
        unlabel_batch = output.size()[0]

        for ind1 in range(unlabel_batch - 5):  # try to make every sample contribute
            # randomly pick two other samples
            ind2 = random.randint(ind1 + 2, unlabel_batch - 1)  # big distance
            ind3 = random.randint(ind1 + 1, ind2 - 1)  # small distance

            # target1 = Variable(x_encode[ind2,:].data, requires_grad=False).cuda()
            # target2 = Variable(x_encode[ind3,:].data, requires_grad=False).cuda()
            # diff_big = criterion(x_encode[ind1,:], target1)
            # diff_small = criterion(x_encode[ind1,:], target2)
            # import ipdb; ipdb.set_trace() ?

            diff_big = torch.sum((output[ind1] - output[ind2]) ** 2) / 2.0
            diff_small = torch.sum((output[ind1] - output[ind3]) ** 2) / 2.0

            loss_unlabel += diff_small - Thresh - diff_big

        return loss_unlabel

    def forward_unlabel(self, sample):
        """
        :param sample: unlabeled data
        :return: unlabel loss
        """
        inputValue = sample.squeeze().to(self.device)
        output = self.model(inputValue)

        loss = self.unlabel_loss(output)
        return loss

    def forward_label(self, sample):
        """
        :param sample: labeled data
        :return: label loss
        """
        inputValue = sample['img'].to(self.device)
        targetValue = sample['label'].to(self.device)

        output = self.model(inputValue)

        loss = self.criterion(output, targetValue)
        return loss

    def visualize_output(self, inputs, outputs):
        seq_show_with_arrow(inputs.cpu().numpy(), outputs.detach().cpu().numpy(),
                            scale=0.8, mean=mean, std=std)

    def test_label(self, val_sample, visualize):
        """ """
        inputImgs = val_sample['img'].to(self.device)
        labels = val_sample['label'].to(self.device)

        output = self.model(inputImgs)
        loss_label = self.criterion(output, labels)

        # import ipdb;ipdb.set_trace()
        if visualize:
            self.visualize_output(inputImgs, output)

            angle_error, cls_accuracy = self.angle_metric(
                output.detach().cpu().numpy(), labels.cpu().numpy())

            print 'label-loss %.4f, angle diff %.4f, accuracy %.4f' % (loss_label.item(), angle_error, cls_accuracy)

        return loss_label

    def test_unlabel(self, val_sample, visualize):
        """ """
        inputImgs = val_sample.squeeze().to(self.device)
        output = self.model(inputImgs)
        loss_unlabel = self.unlabel_loss(output)

        # import ipdb;ipdb.set_trace()
        if visualize:
            self.visualize_output(inputImgs, output)

            print loss_unlabel.item()

        return loss_unlabel

    def test_label_unlabel(self, val_sample, visualize):
        """ """
        inputImgs = val_sample['imgseq'].squeeze().to(self.device)
        labels = val_sample['labelseq'].squeeze().to(self.device)

        output = self.model(inputImgs)
        loss_label = self.criterion(output, labels)

        loss_unlabel = self.unlabel_loss(output)
        loss = loss_label + Lamb * loss_unlabel

        # import ipdb;ipdb.set_trace()
        if visualize:
            self.visualize_output(inputImgs, output)

            angle_error, cls_accuracy = self.angle_metric(
                output.detach().cpu().numpy(), labels.cpu().numpy())

            print 'loss %.4f, label-loss %.4f, unlabel-loss %.4f, angle diff %.4f, accuracy %.4f' % (loss.item(),
                                                                                                     loss_label.item(), loss_unlabel.item(), angle_error, cls_accuracy)

        return loss, loss_label, loss_unlabel

    def train(self):
        """ train model (one batch) """
        super(MyWF, self).train()

        self.countTrain += 1
        self.model.train()

        # get next labeled sample
        sample, self.train_data_iter, self.labelEpoch = self.next_sample(
            self.train_data_iter, self.train_loader, self.labelEpoch)

        # get next unlabeled sample
        sample_unlabel, self.train_unlabel_data_iter, self.unlabelEpoch = self.next_sample(
            self.train_unlabel_data_iter, self.train_unlabel_loader, self.unlabelEpoch)

        # calculate loss
        label_loss = self.forward_label(sample)
        unlabel_loss = self.forward_unlabel(sample_unlabel)
        loss = label_loss + Lamb * unlabel_loss

        # backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update training loss history
        self.AV['loss'].push_back(loss.item())
        self.AV['label_loss'].push_back(label_loss.item())
        self.AV['unlabel_loss'].push_back(unlabel_loss.item())

        # record current params
        if self.countTrain % ShowIter == 0:
            loss_str = self.get_log_str()
            self.logger.info("%s #%d - (%d %d) %s lr: %.6f" % (exp_prefix[:-1],
                                                               self.countTrain, self.labelEpoch, self.unlabelEpoch, loss_str, learn))
        # save temporary model
        if (self.countTrain % Snapshot == 0):
            self.save_snapshot()

    def test(self, visualize=False):
        """ test model (one batch) """

        # put into test mode
        super(MyWF, self).test()
        self.model.eval()

        # get next sample
        sample, self.test_data_iter, self.testEpoch = self.next_sample(
            self.test_data_iter, self.test_loader, self.testEpoch)

        # calculate test loss
        if TestType == 1 or TestType == 0:  # labeled sequence
            loss, loss_label, loss_unlabel = self.test_label_unlabel(
                sample, visualize)

        elif TestType == 2:  # labeled folder
            loss_label = self.test_label(sample, visualize)

        elif TestType == 3:  # unlabeled sequence
            loss_unlabel = self.test_unlabel(sample, visualize)

        # update test loss history
        if TestType == 0:  # why only 0 ?
            self.AV['test_loss'].push_back(loss.item(), self.countTrain)
            self.AV['test_label'].push_back(loss_label.item(), self.countTrain)
            self.AV['test_unlabel'].push_back(
                loss_unlabel.item(), self.countTrain)

    def train_all():
        # the logic is not yet consistent ?

        for iteration in range(Trainstep):
            wf.train()

            if TestType > 0:    # ?
                wf.test(visualize=True)
            elif iteration % TestIter == 0:
                wf.test()


def main():
    """ Train and validate new model """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # train and validate
    try:
        # Instantiate an object for MyWF.
        wf = MyWF("./", prefix=exp_prefix).
        wf.initialize(device)

        wf.train_all()

        wf.finalize()

    except WorkFlow.SigIntException as e:
        wf.finalize()
    except WorkFlow.WFException as e:
        print(e.describe())

if __name__ == "__main__":
    main()
