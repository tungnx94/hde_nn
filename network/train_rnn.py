import sys
sys.path.insert(0, "..")

import os
import ipdb
import network

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger

from torch.autograd import Variable
from RNNModels import MobilenetGRU

from dataset.generalData import DataLoader
from dataset.dukeSeqLabelData import DukeSeqLabelDataset
from dataset.viratSeqLabelData import ViratSeqLabelDataset

from utils import img_denormalize, put_arrow

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


TrainAnnotations = 'DukeMCMT/trainval_duke.txt'
ValAnnotations = 'DukeMCMT/test_heading_gt.txt'
ImageFolder = 'DukeMCMT/heading/' # heading_zip ?
LogFolder = '../outputs/tf_logs/'
ModelFolder = '../outputs/intermediate_models/'

class Training:

    def __init__(self, lr=1e-4, expt_name='part1_1', remove_all_log=True,
                 load_previous=None, iter_bias=0, batch_size=1,
                 seq_batch_size=6,
                 train_annotations_filename=TrainAnnotations,
                 val_annotations_filename=ValAnnotations,
                 tf_logs_folder=LogFolder,
                 intermediate_models_folder=ModelFolder,
                 img_size=192, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 test_freq=5000, model_grad_log=10000, log_loss=20, save_net_freq=10000,
                 data_aug=True):

        self.model = MobilenetGRU()
        # network.weights_xavier_init(self.model)
        self.params = list(self.model.parameters())
        # ipdb.set_trace()
        self.lr = lr
        #self.optimizer = torch.optim.Adam(self.params, lr=lr)
        self.optimizer = torch.optim.SGD(self.params, lr=lr, momentum=0.9)
        # self.criterion = nn.CrossEntropyLoss()#nn.BCELoss()
        self.criterion = nn.MSELoss()

        if IS_CUDA:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.test_freq = test_freq
        self.model_grad_log = model_grad_log
        self.log_loss = log_loss
        self.save_net_freq = save_net_freq
        self.batch_size = batch_size
        self.expt_name = expt_name
        self.iter_bias = iter_bias
        self.load_previous = load_previous
        self.intermediate_models_folder = intermediate_models_folder
        self.mean = mean
        self.std = std

        # load model from checkpoint
        if self.load_previous is not None:
            self.iter_bias, self.model, self.optimizer = network.load_checkpoint(
                self.load_previous, self.model, self.optimizer)
            remove_all_log = False

        # logger
        self.tf_logger = Logger(
            tf_logs_folder, name=expt_name, remove_all_log=remove_all_log)

        """
        self.train_dataset = DukeSeqLabelDataset(label_file=train_annotations_filename,
                                            seq_length=seq_batch_size,
                                            img_size=img_size,
                                            mean=mean, std=std,
                                            data_aug=data_aug)
        """
        # using outdated version of Dataset ? role of images_root_folder ?
        self.train_dataset = ViratSeqLabelDataset(label_file=train_annotations_filename,
                                                  seq_length=seq_batch_size,
                                                  img_size=img_size,
                                                  mean=mean, std=std,
                                                  data_aug=data_aug)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, num_workers=4)

        if val_annotations_filename is not None:
            """
            self.val_dataset = DukeSeqLabelDataset(label_file=val_annotations_filename,
                                    seq_length=seq_batch_size,
                                    img_size=img_size,
                                    mean=mean, std=std,
                                    data_aug=data_aug)
            """
            self.val_dataset = ViratSeqLabelDataset(label_file=val_annotations_filename,                   
                                                    seq_length=seq_batch_size,
                                                    img_size=img_size,
                                                    mean=mean, std=std,
                                                    data_aug=data_aug)

            self.val_loader = DataLoader(self.val_dataset, num_workers=4, batch_size=batch_size)

    def forward_pass(self, input):
        imgseq = input['imgseq'].squeeze()
        labelseq = input['labelseq'].squeeze()
        if IS_CUDA:
            imgseq = imgseq.cuda()
            labelseq = labelseq.cuda()
        imgseq = Variable(imgseq, requires_grad=True)
        labelseq = Variable(labelseq, requires_grad=False)

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
            losses = AverageMeter() # loss history

            for i1, input in enumerate(self.train_loader):

                output, labelseq = self.forward_pass(input)
                loss = self.criterion(output, labelseq)
                #loss = loss/self.batch_size

                # backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.update(loss.data[0], 1)

                if i1 % 100 == 0:
                    print('\t%d,Loss:%.3f' % (i1, loss.data[0]))

                step += 1
                # test model
                if (step - 1) % self.test_freq == 0:
                    self.validate(step, self.tf_logger)
                    self.model.train() #put back to training mode

                if (step - 1) % self.model_grad_log == 0:
                    self.tf_logger.model_param_histo_summary(
                        self.model, step, postfix='')

                if (step - 1) % self.log_loss == 0:
                    self.tf_logger.scalar_summary(
                        'train/loss', loss.data[0], step)

                # save intermediate model
                if step % self.save_net_freq == 0 or (step == 1 and self.load_previous is None):
                    save_name = os.path.join(
                        self.intermediate_models_folder, '{}_{}.pth.tar'.format(self.expt_name, step))

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

    # lr=0.0001
    lr = 1e-5

    #dropout = 0.2
    test_freq = 1000
    model_grad_log = 2000
    log_loss = 20
    save_net_freq = 10000

    train_annotations_filename = 'VIRAT/train/annotations/annotations.csv'
    val_annotations_filename = 'VIRAT/val/annotations/annotations.csv'

    expt_name = 'virat_mobilenet_gru_w_logit_no_pool_sgd_lr%s' % (str(lr))

    training = Training(lr=lr, load_previous=load_previous, expt_name=expt_name,
                        train_annotations_filename=train_annotations_filename,
                        val_annotations_filename=val_annotations_filename,
                        test_freq=test_freq, model_grad_log=model_grad_log, log_loss=log_loss, save_net_freq=save_net_freq
                        )

    # ipdb.set_trace()
    training.train(num_iters=5000)

if __name__ == '__main__':
    main()
