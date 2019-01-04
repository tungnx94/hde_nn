# This file is modified from tran_ed_semi_cls.py
# Change the classification model to regression model
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from os.path import join as joinPath
from dataset.generalData import DataLoader

from utils import get_path, groupPlot
from network import EncoderReg_Pred
from dataset import TrackingLabelDataset, FolderLabelDataset, FolderUnlabelDataset, DukeSeqLabelDataset

UseGPU = torch.cuda.is_available()

exp_prefix = '20_2_'

OutDir = './log/test/cpu'


ModelFile = joinPath(OutDir, "loss.png")

LossFile = joinPath(OutDir, exp_prefix + 'lossplot.npy')
ValLossFile = joinPath(OutDir, exp_prefix + 'vallossplot.npy')
LabelLossFile = joinPath(OutDir, exp_prefix + 'unlabellossplot.npy')
UnlabelLossFile = joinPath(OutDir, exp_prefix + 'labellossplot.npy')

DukeLabelFile = get_path("DukeMCMT/trainval_duke.txt")
HandLabelFolder = get_path("label")

UnlabelFolder = get_path("dirimg")

TestLabelFile = get_path("DukeMCMT/test_heading_gt.txt")
TestLabelFolder = get_path("val_drone")

LR = 0.01
Lamb = 0.5

TrainBatch = 32
UnlabelBatch = 32  # sequence length
ValBatch = 100

TrainStep = 200  # 10000
ShowIter = 10
SnapShot = 50  # 500
TrainLayers = 0

Hiddens = [3, 16, 32, 32, 64, 64, 128, 256]
Kernels = [4, 4, 4, 4, 4, 4, 3]
Paddings = [1, 1, 1, 1, 1, 1, 0]
Strides = [2, 2, 2, 2, 2, 2, 1]


def visualize(lossplot, labellossplot, unlabellossplot, vallossplot):
    labellossplot = np.array(labellossplot).reshape((-1, 1)).mean(axis=1)
    vallossplot = np.array(vallossplot)

    # figure 1
    ax1 = plt.subplot(131)
    ax1.plot(labellossplot)
    ax1.plot(vallossplot)
    ax1.grid()

    # figure 2
    lossplot = np.array(lossplot).reshape((-1, 1)).mean(axis=1)
    ax2 = plt.subplot(132)
    ax2.plot(lossplot)
    ax2.grid()

    # firgure 3
    unlabellossplot = np.array(unlabellossplot)
    gpunlabelx, gpunlabely = groupPlot(
        range(len(unlabellossplot)), unlabellossplot)

    ax3 = plt.subplot(133)
    ax3.plot(unlabellossplot)
    ax3.plot(gpunlabelx, gpunlabely, color='y')
    ax3.grid()

    plt.savefig(ModelFile)


def train_label_unlabel(encoderReg, sample, unlabel_sample, optimizer, criterion, lamb):
    """ train one step """
    # label
    inputImgs = sample['img']
    labels = sample['label']

    inputState = encoderReg.new_variable(inputImgs, requires_grad=True)
    targetreg = encoderReg.new_variable(labels, requires_grad=False)

    # unlabel
    imgseq = unlabel_sample.squeeze()
    inputState_unlabel = encoderReg.new_variable(imgseq, requires_grad=True)

    # forward pass
    output, _, _ = encoderReg(inputState)

    output_unlabel, encode, pred = encoderReg(inputState_unlabel)
    pred_target = encode[UnlabelBatch / 2:, :].detach()  # ?

    loss_label = criterion(output, targetreg)
    loss_pred = criterion(pred, pred_target)  # unlabel loss

    loss = loss_label + loss_pred * Lamb

    # backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss_label.item(), loss_pred.item(), loss.item()


def test_label(val_sample, encoderReg, criterion, batchnum=1):
    """ validate on labeled dataset """
    inputImgs = val_sample['img']
    labels = val_sample['label']
    inputState = encoderReg.new_variable(inputImgs, requires_grad=False)
    targetreg = encoderReg.new_variable(labels, requires_grad=False)

    output, _, _ = encoderReg(inputState)
    loss = criterion(output, targetreg)

    return loss.item()


def save_snapshot(model, ind, label_loss, unlabel_loss, total_loss, val_loss):
    file = joinPath(OutDir, 'model_' + str(ind) + '.pkl')
    torch.save(model.state_dict(), file)

    np.save(LossFile, total_loss)
    np.save(ValLossFile, val_loss)
    np.save(LabelLossFile, label_loss)
    np.save(UnlabelLossFile, unlabel_loss)


def main():
    start_t = datetime.now()

    # Encoder model
    encoderReg = EncoderReg_Pred(
        Hiddens, Kernels, Strides, Paddings, actfunc='leaky', rnnHidNum=128, device="cuda")

    encoderReg.load_to_device()

    paramlist = list(encoderReg.parameters())
    regOptimizer = optim.SGD(paramlist[-TrainLayers:], lr=LR, momentum=0.9)
    # regOptimizer = optim.Adam(paramlist[-TrainLayers:], lr = lr)

    criterion = nn.MSELoss()

    # Datasets
    print "loading datasets"
    imgdataset = TrackingLabelDataset("duke",
                                      data_file=DukeLabelFile, data_aug=True)  # Duke, 225426
    imgdataset2 = FolderLabelDataset("handlabel",
                                     img_dir=HandLabelFolder, data_aug=True)  # HandLabel, 1201

    unlabelset = FolderUnlabelDataset("ucf",
                                      img_dir=UnlabelFolder, seq_length=UnlabelBatch, data_aug=True, extend=True)

    valset = FolderLabelDataset("test", TestLabelFolder, data_aug=False)
    #valset2 = DukeSeqLabelDataset(TestLabelFolder, data_aug=False)

    # Dataloaders
    dataloader = DataLoader(imgdataset, batch_size=TrainBatch, num_workers=2)
    dataloader2 = DataLoader(imgdataset2, batch_size=TrainBatch, num_workers=2)

    unlabelloader = DataLoader(unlabelset, num_workers=2)

    valloader = DataLoader(valset, batch_size=ValBatch,
                           num_workers=2, shuffle=False)

    # Loss history
    lossplot = []
    labellossplot = []
    unlabellossplot = []
    vallossplot = []

    # Train
    val_loss = 0.0
    for ind in range(1, TrainStep + 1):
        # load next samples
        if ind % 2 == 0:
            sample = dataloader.next_sample()
        else:
            sample = dataloader2.next_sample()

        unlabel_sample = unlabelloader.next_sample()

        # run one training step
        label_loss, unlabel_loss, total_loss = train_label_unlabel(
            encoderReg, sample, unlabel_sample, regOptimizer, criterion, Lamb)

        labellossplot.append(label_loss)
        unlabellossplot.append(unlabel_loss)
        lossplot.append(total_loss)

        # Validate on test set
        if ind % ShowIter == 0:
            val_losses = [test_label(val_sample, encoderReg, criterion)
                          for val_sample in valloader]
            val_loss = sum(val_losses) / len(val_losses)  # take average

            vallossplot.append(val_loss)

            print("[%s %d] loss: %.5f, label: %.5f, unlabel: %.5f, val: %.5f " % (
                exp_prefix[:-1], ind, total_loss, label_loss, unlabel_loss, val_loss))

        if ind % SnapShot == 0:  # Save model + loss
            save_snapshot(encoderReg, ind, labellossplot,
                          unlabellossplot, lossplot, vallossplot)

    visualize(lossplot, labellossplot, unlabellossplot, vallossplot)

    end_t = datetime.now()

    print "Training completed"
    print "Elapsed time: {}".format(end_t - start_t)

if __name__ == "__main__":
    main()
