import os
import network

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network import MobilenetGRU
from utils import Logger, img_denormalize, put_arrow, new_variable, get_path
from dataset import DataLoader, DukeSeqLabelDataset, ViratSeqLabelDataset

IS_CUDA = torch.cuda.is_available()  # True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


TrainFile = get_path('DukeMCMT/trainval_duke.txt')
ValFile = get_path('DukeMCMT/test_heading_gt.txt')
ImageFolder = get_path('DukeMCMT/heading')  # heading_zip ?
LogFolder = 'logdata'
ModelFolder = 'models'

Lr = 1e-4  # 1e-5
TestFreq = 200  # 5000
SaveFreq = 500 # 10000
LossPrintFreq = 100
ImgSumFreq = 50
BatchSize = 1
SeqLength = 6
ImgSize = 192

LogLossFreq = 20
LogGradFreq = 10000  # 2000

PretrainModel = None

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

ExptName = 'part1_1'


class Training:

    def __init__(self, pre_model=None, remove_all_log=True):

        self.model = MobilenetGRU()
        self.params = list(self.model.parameters())
        self.pre_model = pre_model

        #self.optimizer = torch.optim.Adam(self.params, lr=lr)
        self.optimizer = torch.optim.SGD(self.params, lr=Lr, momentum=0.9)

        # load model from checkpoint
        if pre_model is not None:
            self.current_iter, self.model, self.optimizer = network.load_checkpoint(
                pre_model, self.model, self.optimizer)
            remove_all_log = False
        else:
            self.current_iter = 0

        # self.criterion = nn.CrossEntropyLoss()#nn.BCELoss()
        self.criterion = nn.MSELoss()

        if IS_CUDA:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        # logger
        self.tf_logger = Logger(LogFolder, name=ExptName, remove_all_log=remove_all_log)

        # Dataset & loader
        self.train_dataset = DukeSeqLabelDataset(label_file=TrainFile, seq_length=SeqLength,
                                                 img_size=ImgSize, mean=MEAN, std=STD, data_aug=True)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=BatchSize, num_workers=4)

        self.val_dataset = DukeSeqLabelDataset(label_file=ValFile, seq_length=SeqLength,
                                               img_size=ImgSize, mean=MEAN, std=STD, data_aug=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=BatchSize, num_workers=4)

    def forward_pass(self, sample):
        imgseq = new_variable(sample['imgseq'].squeeze(), requires_grad=True)
        labelseq = new_variable(
            sample['labelseq'].squeeze(), requires_grad=False)

        output = self.model(imgseq)
        output = output.view(output.size(0), -1)

        return imgseq, output, labelseq

    def train(self, epochs=10, num_iters=200):
        """
        num_iters: #epochs
        """
        self.model.train()
        step = self.current_iter
        iter_count = 0

        for i in range(epochs):
            losses = AverageMeter()  # loss history

            for i1, sample in enumerate(self.train_loader):

                _, output, labelseq = self.forward_pass(sample)
                loss = self.criterion(output, labelseq)

                # backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.update(loss.item(), 1)

                if i1 % LossPrintFreq == 0:
                    print('\t%d,Loss:%.3f' % (i1, loss.item()))

                step += 1
                # validate model
                if step % TestFreq == 0:
                    self.validate(step)
                    self.model.train()  # back to training mode

                if step % LogGradFreq == 0:
                    self.tf_logger.model_param_histo_summary(
                        self.model, step, postfix='')

                if step % LogLossFreq == 0:
                    self.tf_logger.scalar_summary(
                        'train/loss', loss.item(), step)

                # save intermediate model
                if step % SaveFreq == 0 or (step == 1 and self.pre_model is None):
                    save_name = os.path.join(
                        ModelFolder, '{}_{}.pth.tar'.format(ExptName, step))

                    network.save_checkpoint({
                        'epoch': step + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, filename=save_name)

                    print('Saved model to {}'.format(save_name))

                iter_count += 1
                if iter_count >= num_iters:
                    return

    def gen_tf_image_summary(self, imgs, outs):
        ret_img = np.zeros_like(imgs).transpose((0, 2, 3, 1))
        for i, (img, out) in enumerate(zip(imgs, outs)):

            img = img_denormalize(img, mean=MEAN, std=STD)
            img = put_arrow(img, out)
            ret_img[i, ...] = img

        return ret_img

    def validate(self, step, num_iters=250):
        self.model.eval()
        losses = AverageMeter()
        angle_diff = AverageMeter()

        for i1, sample in enumerate(self.val_loader):
            if i1 > num_iters:
                break

            imgseq, output, labelseq = self.forward_pass(sample)
            loss = self.criterion(output, labelseq)

            output_data = output.data.cpu().numpy()
            labelseq_data = labelseq.data.cpu().numpy()

            angle_diff_val = np.mean(np.abs(np.arctan2(output_data[:, 0], output_data[
                                     :, 1]) - np.arctan2(labelseq_data[:, 0], labelseq_data[:, 1])))

            if i1 % ImgSumFreq == 0:
                img_summary = self.gen_tf_image_summary(
                    imgseq.data.cpu().numpy()[:10], output.data.cpu().numpy()[:10])

                self.tf_logger.image_summary(
                    'val/%d-%d' % (step, i1), img_summary, step)

            losses.update(loss.item(), 1)
            angle_diff.update(angle_diff_val, 1)

        self.tf_logger.scalar_summary('val/loss', losses.avg, step)
        self.tf_logger.scalar_summary('val/angle_diff', angle_diff_val, step)


def main():
    training = Training()
    training.train(num_iters=1000)

if __name__ == '__main__':
    main()
