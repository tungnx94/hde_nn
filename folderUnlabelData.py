# extend with UCF data

import cv2
import os
import pickle
import random

import numpy as np
from os.path import join

from generalData import SingleDataset


class FolderUnlabelDataset(SingleDataset):

    def __init__(self, img_dir='/datadrive/person/dirimg/', data_file='',
                 img_size=192, data_aug=False, mean=[0, 0, 0], std=[1, 1, 1],
                 batch=32, extend=False, include_all=False):

        super(FolderUnlabelDataset, self)__init__(img_size, data_aug, mean, 0, std)

        self.batch = batch
        self.img_seqs = []  # image sequence list
        self.episodes = []  # episode numbers

        # load from saved pickle file
        if data_file != '':
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            self.N = data['N']
            self.episodes = data['episodeNum']
            self.img_seqs = data['img_seqs']
            return

        self.load_image_sequences()

        # calculate episode length accumulative
        total_seq_num = 0
        for seq in self.img_seqs:
            total_seq_num += len(seq) - batch + 1
            self.episodes.append(total_seq_num)
 
        self.N = total_seq_num

        # import ipdb; ipdb.set_trace()

        # Save loaded data for future use
        if data_file == '':
            with open('unlabeldata.pkl', 'wb') as f:
                pickle.dump({'N': self.N, 'episodeNum': self.episodes,
                             'img_seqs': self.img_seqs}, f, pickle.HIGHEST_PROTOCOL)

        # debug
        print 'Read #sequences: ', len(self.img_seqs)
        print np.sum(np.array([len(img_list) for img_list in self.img_seqs]))

    def load_image_sequences(self):
        # img_folders = ['4','7','11','17','23','30','32','33','37','38','49','50','52']
        img_folders = []
        if include_all:  # include all the folders in one directory -- for duke
            img_folders = os.listdir(img_dir)
        elif extend:
            img_folders = [str(k) for k in range(101, 1040)]

        # process each folder
        for folder in img_folders:
            folder_path = join(img_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            # all images in this folder
            img_list = sorted(os.listdir(folder_path))

            sequence = []
            last_idx = -1

            for file_name in img_list:
                if not file_name.endswith(".jpg"):  # only process jpg
                    continue

                file_path = join(folder_path, file_name)

                if include_all:  # duke dataset
                    sequence.append(file_path)
                    continue

                # filtering the incontineous data
                file_idx = file_name.split('.')[0].split('_')[-1]
                try:
                    file_idx = int(file_idx)
                except:
                    print 'filename parse error:', file_name, file_idx
                    continue

                if last_idx < 0 or file_idx == last_idx + 1: # continuous index
                    sequence.append(file_path)
                    last_idx = file_idx
                else:  # indexes not continuous
                    # save sequence if long enough
                    if len(sequence) >= batch:
                        self.img_seqs.append(sequence)
                        sequence = []

                    last_idx = -1

            # save if long enough
            if len(sequence) >= batch:
                # print '*** sequence: ', len(sequence)
                self.img_seqs.append(sequence)
                sequence = []

    def __getitem__(self, idx):
        ep_idx = 0  # calculate the episode index
        while idx >= self.episodes[ep_idx]:
            ep_idx += 1

        if ep_idx > 0:
            idx -= self.episodes[ep_idx - 1]

        # random flip all images in batch
        flipping = self.get_flipping()

        # print ep_idx, idx
        imgseq = []
        for k in range(self.batch):
            img = cv2.imread(self.img_seqs[ep_idx][idx + k])

            out_img, _ = self.get_img_and_label(img, None, flipping)
            imgseq.append(out_img)

        return np.array(imgseq)


def main():
    # test
    from torch.utils.data import DataLoader
    from utils import seq_show

    np.set_printoptions(precision=4)

    # unlabelset = FolderUnlabelDataset(img_dir='/datadrive/person/dirimg',batch = 24, extend=True, data_aug=True)#,data_file='duke_unlabeldata.pkl')
    # unlabelset = FolderUnlabelDataset(batch=24, data_aug=True, extend=True, data_file='drone_ucf_unlabeldata.pkl')
    unlabelset = FolderUnlabelDataset(
        img_dir='/home/wenshan/headingdata/DukeMCMT/heading', batch=24, data_aug=True, include_all=True)

    dataloader = DataLoader(unlabelset, batch_size=1,
                            shuffle=True, num_workers=1)
    data_iter = iter(dataloader)

    while True:
        try:
            sample = data_iter.next()
        except:
            data_iter = iter(dataloader)
            sample = data_iter.next()

        seq_show(sample.squeeze().numpy(), scale=0.8)

    """
    print len(unlabelset)
    for k in range(1):
        imgseq = unlabelset[k * 1000]
        print imgseq.dtype, imgseq.shape
        seq_show(imgseq)
    """

if __name__ == '__main__':
    main()
