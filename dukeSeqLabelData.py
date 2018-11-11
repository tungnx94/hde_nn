# for dukeMCMT dataset
# return a sequence of data with label

import os
import cv2
import numpy as np

from utils import im_scale_norm_pad, img_denormalize, seq_show, im_hsv_augmentation, im_crop
import random


from generalData import SingleDataset


class DukeSeqLabelDataset(SingleDataset):

    def __init__(self, label_file='/datadrive/person/DukeMTMC/heading_gt.txt',
                 img_size=192, data_aug=False, mean=[0, 0, 0], std=[1, 1, 1], batch=32):

        super(DukeSeqLabelDataset, self)__init__(img_size, data_aug, mean, 0, std)

        self.batch = batch
        self.img_seqs = []
        self.episodes = []

        frame_iter = 6

        sequencelist = []
        img_dir = os.path.split(label_file)[0] # correct 
        
        # images
        with open(label_file, 'r') as f:
            lines = f.readlines()

        lastind = -1
        lastcam = -1
        for line in lines:
            [img_name, angle] = line.strip().split(' ')
            frameid = img_name.strip().split('/')[-1].split('_')[1][5:]
            try:
                frameid = int(frameid)
            except:
                print 'filename parse error:', img_name, frameid
                continue

            file_path = os.path.join(img_dir, img_name)
            camnum = img_name.strip().split('_')[0]

            # import ipdb; ipdb.set_trace()
            if (lastind < 0 or frameid == lastind + frame_iter) and (camnum == lastcam or lastcam == -1):
                sequencelist.append((file_path, angle))
                lastind = frameid
                lastcam = camnum
            else:  # the index is not continuous
                if len(sequencelist) >= batch:
                    self.img_seqs.append(sequencelist)
                    print '** sequence: ', len(sequencelist)
                    sequencelist = []
                else:
                    print 'sequence too short'
                lastind = -1
                lastcam = -1

        # total length
        total_seq_num = 0
        for sequ in self.img_seqs:
            total_seq_num += len(sequ) - batch + 1
            self.episodes.append(total_seq_num)
        self.N = total_seq_num

        # debug
        print 'Read #sequences: ', len(self.img_seqs)
        print 'Read #images: ', sum([len(sequence) for sequence in self.img_seqs])

    def __getitem__(self, idx):
        epiInd = 0  # calculate the epiInd
        while idx >= self.episodes[epiInd]:
            # print self.episodes[epiInd],
            epiInd += 1
        if epiInd > 0:
            idx -= self.episodes[epiInd - 1]

        # random fliping
        flipping = self.get_flipping()

        # print epiInd, idx
        imgseq = []
        labelseq = []
        for k in range(self.batch):
            img = cv2.imread(self.img_seqs[epiInd][idx + k][0])
            angle = self.img_seqs[epiInd][idx + k][1]

            direction_angle_cos = np.cos(float(angle))
            direction_angle_sin = np.sin(float(angle))
            label = np.array(
                [direction_angle_sin, direction_angle_cos], dtype=np.float32)

            if self.aug:
                img = im_hsv_augmentation(img)
                img = im_crop(img)
                if flipping:
                    label[1] = - label[1]

            outimg = im_scale_norm_pad(
                img, outsize=self.img_size, mean=self.mean, std=self.std, down_reso=True, flip=flipping)

            imgseq.append(outimg)
            labelseq.append(label)

        return {'imgseq': np.array(imgseq), 'labelseq': np.array(labelseq)}


def unlabelloss(labelseq):
    thresh = 0.005
    unlabel_batch = labelseq.shape[0]
    loss_unlabel = 0
    for ind1 in range(unlabel_batch - 5):  # try to make every sample contribute
        # randomly pick two other samples
        ind2 = random.randint(ind1 + 2, unlabel_batch - 1)  # big distance
        ind3 = random.randint(ind1 + 1, ind2 - 1)  # small distance

        # target1 = Variable(x_encode[ind2,:].data, requires_grad=False).cuda()
        # target2 = Variable(x_encode[ind3,:].data, requires_grad=False).cuda()
        # diff_big = criterion(x_encode[ind1,:], target1)
        # #(labelseq[ind1]-labelseq[ind2])*(labelseq[ind1]-labelseq[ind2])
        diff_big = (labelseq[ind1] - labelseq[ind2]) * \
            (labelseq[ind1] - labelseq[ind2])
        diff_big = diff_big.sum() / 2.0
        # diff_small = criterion(x_encode[ind1,:], target2)
        # #(labelseq[ind1]-labelseq[ind3])*(labelseq[ind1]-labelseq[ind3])
        diff_small = (labelseq[ind1] - labelseq[ind3]) * \
            (labelseq[ind1] - labelseq[ind3])
        diff_small = diff_small.sum() / 2.0
        # import ipdb; ipdb.set_trace()
        cost = max(diff_small - thresh - diff_big, 0)
        # print diff_big, diff_small, cost
        loss_unlabel = loss_unlabel + cost
    print loss_unlabel


def main():
    # test
    from torch.utils.data import DataLoader

    np.set_printoptions(precision=4)

    # unlabelset = FolderUnlabelDataset(img_dir='/datadrive/person/DukeMTMC/heading',batch = 32, data_aug=True, include_all=True,datafile='duke_unlabeldata.pkl')
    # unlabelset = FolderUnlabelDataset(img_dir='/datadrive/person/DukeMTMC/heading',batch = 24, data_aug=True, include_all=True)
    unlabelset = DukeSeqLabelDataset(
        label_file='/datadrive/person/DukeMTMC/test_heading_gt.txt', batch=24, data_aug=True)
    print len(unlabelset)

    dataloader = DataLoader(unlabelset, batch_size=1,
                            shuffle=True, num_workers=1)
    dataiter = iter(dataloader)
    while True:

        try:
            sample = dataiter.next()
        except:
            dataiter = iter(dataloader)
            sample = dataiter.next()

        imgseq, labelseq = sample['imgseq'].squeeze().numpy(), sample[
            'labelseq'].squeeze().numpy()
        unlabelloss(labelseq)
        fakelabel = np.random.rand(24, 2)
        unlabelloss(fakelabel)
        seq_show(imgseq, dir_seq=labelseq)

    # import ipdb; ipdb.set_trace()

    # for k in range(10):
    #     sample = unlabelset[k*1000]
    #     imgseq, labelseq = sample['imgseq'], sample['labelseq']
    #     print imgseq.dtype, imgseq.shape
    #     seq_show_with_arrow(imgseq,labelseq, scale=0.8)

if __name__ == '__main__':
    main()
