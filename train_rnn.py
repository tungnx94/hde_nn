import os
import network

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger

from RNNModels import MobilenetGRU
from utils import img_denormalize, put_arrow, new_variable
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


TrainFile = 'DukeMCMT/trainval_duke.txt'
ValFile = 'DukeMCMT/test_heading_gt.txt'
ImageFolder = 'DukeMCMT/heading/'  # heading_zip ?
LogFolder = 'outputs/tf_logs/'
ModelFolder = 'outputs/intermediate_models/'

Lr = 1e-4  # 1e-5
TestFreq = 5000  # 1000
SaveFreq = 10000 #
BatchSize = 1 #
SeqLength = 6 #
ImgSize = 192 #

LogLossFreq = 20 #
LogGradFreq = 10000 # 2000

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

expt_name = 'part1_1'


class Training:

    def __init__(self, remove_all_log=True, load_previous=None):

        self.model = MobilenetGRU()
        self.params = list(self.model.parameters())

        #self.optimizer = torch.optim.Adam(self.params, lr=lr)
        self.optimizer = torch.optim.SGD(self.params, lr=Lr, momentum=0.9)

        # self.criterion = nn.CrossEntropyLoss()#nn.BCELoss()
        self.criterion = nn.MSELoss()

        if IS_CUDA:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()f

        self.iter_bias = 0
        self.load_previous = load_previous

        # load model from checkpoint
        if self.load_previous is not None:
            self.iter_bias, self.model, self.optimizer = network.load_checkpoint(
                self.load_previous, self.model, self.optimizer)
            remove_all_log = False

        # logger
        self.tf_logger = Logger(LogFolder, name=expt_name, remove_all_log=remove_all_log)

        # Dataset & loader
        self.train_dataset = DukeSeqLabelDataset(label_file=TrainFile, seq_length=SeqLength,
                                                 img_size=ImgSize, mean=mean, std=std, data_aug=True)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=4)

        self.val_dataset = DukeSeqLabelDataset(label_file=ValFile, seq_length=SeqLength,
                                               img_size=ImgSize, mean=MEAN, std=STD, data_aug=True)

        self.val_loader = DataLoader(
            self.val_dataset, num_workers=4, batch_size=batch_size)

    def forward_pass(self, input):
        imgseq = new_variable(input['imgseq'].squeeze(), requires_grad=True)
        labelseq = new_variable(
            input['labelseq'].squeeze(), requires_grad=False)

        output = self.model(imgseq)
        output = output.view(output.size(0), -1)

        return output, labelseq

    def train(self, num_iters=200):
        """
        num_iters: #epochs
        """
        self.model.train()
        step = self.iter_bias

        for i in range(num_iters + 1):
            losses = AverageMeter()  # loss history

            for i1, input in enumerate(self.train_loader):

                output, labelseq = self.forward_pass(input)
                loss = self.criterion(output, labelseq)

                # backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.update(loss.data[0], 1)

                if i1 % 100 == 0:
                    print('\t%d,Loss:%.3f' % (i1, loss.data[0]))

                step += 1
                # test model
                if step % TestFreq == 0:
                    self.validate(step, self.tf_logger)
                    self.model.train()  # put back to training mode

                if step % LogGradFreq == 0:
                    self.tf_logger.model_param_histo_summary(
                        self.model, step, postfix='')

                if step % LogLossFreq == 0:
                    self.tf_logger.scalar_summary(
                        'train/loss', loss.data[0], step)

                # save intermediate model
                if step % SaveFreq == 0 or (step == 1 and self.load_previous is None):
                    save_name = os.path.join(
                        ModelFolder, '{}_{}.pth.tar'.format(self.expt_name, step))

                    network.save_checkpoint({
                        'epoch': step + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, filename=save_name)

                    print('Saved model to {}'.format(save_name))
        return

    def gen_tf_image_summary(self, imgs, outs):
        ret_img = np.zeros_like(imgs).transpose((0, 2, 3, 1))
        for i, (img, out) in enumerate(zip(imgs, outs)):

            img = img_denormalize(img, mean=self.mean, std=self.std)
            img = put_arrow(img, out)
            ret_img[i, ...] = img

        return ret_img

    def validate(self, step, logger, num_iters=5000):
        self.model.eval()
        losses = AverageMeter()
        angle_diff = AverageMeter()

        for i1, input in enumerate(self.val_loader):
            # ipdb.set_trace()
            if i1 > num_iters:
                break

            output, labelseq = self.forward_pass(input)
            loss = self.criterion(output, labelseq)

            output_data = output.data.cpu().numpy()
            labelseq_data = labelseq.data.cpu().numpy()

            angle_diff_val = np.mean(np.abs(np.arctan2(output_data[:, 0], output_data[
                                     :, 1]) - np.arctan2(labelseq_data[:, 0], labelseq_data[:, 1])))

            if i1 % 400 == 0:
                img_summary = self.gen_tf_image_summary(
                    imgseq.data.cpu().numpy()[:10], output.data.cpu().numpy()[:10])

                logger.image_summary('val/%d-%d' %
                                     (step, i1), img_summary, step)

            losses.update(loss.data[0], 1)
            angle_diff.update(angle_diff_val, 1)

        logger.scalar_summary('val/loss', losses.avg, step)
        logger.scalar_summary('val/angle_diff', angle_diff_val, step)

        return


def main():
    load_previous = None

    training = Training()

    training.train(num_iters=5000)

if __name__ == '__main__':
    main()
