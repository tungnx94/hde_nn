# extend with UCF data
import sys
sys.path.insert(0, "..")

import os
import cv2
import pickle
import random

import numpy as np
from os.path import join

from generalData import SingleDataset, SequenceDataset


DataFolder = "/home/mohammad/projects/facing_icra/data"


class FolderUnlabelDataset(SequenceDataset):

    def __init__(self, name, img_dir='', data_file=None,
                 img_size=192, data_aug=False, mean=[0, 0, 0], std=[1, 1, 1],
                 seq_length=24, extend=False, include_all=False):

        if data_file != None:
            # load from saved pickle file, priority
            # grandparent
            super(SequenceDataset, self).__init__(
                name, img_size, data_aug, 0, mean, std)
            self.seq_length = seq_length

            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            self.N = data['N']
            self.episodes = data['episodeNum']
            self.img_seqs = data['img_seqs']

            print "{} loaded from saved file".format(self)
        else:
            # load from folder
            self.include_all = include_all
            self.extend = extend
            self.img_dir = img_dir
            # parent
            super(FolderUnlabelDataset, self).__init__(
                name, img_size, data_aug, 0, mean, std, seq_length)

            # Save loaded data for future use
            with open(os.path.join(DataFolder, self.saveName), 'wb') as f:
                pickle.dump({'N': self.N, 'episodeNum': self.episodes,
                             'img_seqs': self.img_seqs}, f, pickle.HIGHEST_PROTOCOL)

            print "{} loaded new".format(self)

        self.read_debug()

    def load_image_sequences(self):
        img_folders = []
        if self.include_all:  # Duke
            img_folders = os.listdir(self.img_dir)
            self.saveName = "duke_unlabeldata.pkl"

        elif self.extend:  # UCF
            img_folders = [str(k) for k in range(101, 1040)]
            self.saveName = "ucf_unlabeldata.pkl"

        # process each folder
        for folder in img_folders:
            folder_path = join(self.img_dir, folder)
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

                if self.include_all:  # duke dataset
                    sequence.append(file_path)
                    continue

                # filtering the incontineous data
                file_idx = file_name.split('.')[0].split('_')[-1]
                try:
                    file_idx = int(file_idx)
                except:
                    print 'filename parse error:', file_name, file_idx
                    continue

                if last_idx < 0 or file_idx == last_idx + 1:  # continuous index
                    sequence.append(file_path)
                    last_idx = file_idx
                else:  # indexes not continuous
                    sequence = self.save_sequence(sequence)
                    last_idx = -1

            # try save
            sequence = self.save_sequence(sequence)

    def __getitem__(self, idx):
        ep_idx, idx = self.get_indexes(idx)
        # random flip all images in seq_length
        flipping = self.get_flipping()

        # print ep_idx, idx
        imgseq = []
        for k in range(self.seq_length):
            img = cv2.imread(self.img_seqs[ep_idx][idx + k])

            out_img, _ = self.get_img_and_label(img, None, flipping)
            imgseq.append(out_img)

        return np.array(imgseq)


def main():
    # test
    from generalData import DataLoader
    from utils import get_path, seq_show

    np.set_printoptions(precision=4)

    duke_img_dir = "DukeMCMT/train"
    ucf_img_dir = "UCF"

    #unlabelset = FolderUnlabelDataset("duke-unlabel", img_dir=get_path(duke_img_dir),
    #                                  seq_length=24, data_aug=True, include_all=True)

    unlabelset = FolderUnlabelDataset("ucf-unlabel", img_dir=get_path(ucf_img_dir),
                                      seq_length=24, data_aug=True, extend=True)

    dataloader = DataLoader(unlabelset)
    count = 5
    for sample in dataloader:
        seq_show(sample.squeeze().numpy(), scale=0.8)

        count -= 1
        if count < 0:
            break

if __name__ == '__main__':
    main()
