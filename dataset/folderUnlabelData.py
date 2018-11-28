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
SaveFile = "unlabeldata.pkl"
SavePath = os.path.join(DataFolder, SaveFile)

class FolderUnlabelDataset(SequenceDataset):

    def __init__(self, img_dir='', data_file='',
                 img_size=192, data_aug=False, mean=[0, 0, 0], std=[1, 1, 1],
                 seq_length=32, extend=False, include_all=False):
        
        if data_file != '':
            # load from saved pickle file, priority
            # grandparent
            super(SequenceDataset, self).__init__(
                img_size, data_aug, 0, mean, std)
            self.seq_length = seq_length

            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            self.N = data['N']
            self.episodes = data['episodeNum']
            self.img_seqs = data['img_seqs']
        else:
            # load from folder
            self.include_all = include_all
            self.extend = extend
            self.img_dir = img_dir
            # parent
            super(FolderUnlabelDataset, self).__init__(
                img_size, data_aug, 0, mean, std, seq_length)

            # Save loaded data for future use
            if data_file == '':
                with open(SavePath, 'wb') as f:
                    pickle.dump({'N': self.N, 'episodeNum': self.episodes,
                                 'img_seqs': self.img_seqs}, f, pickle.HIGHEST_PROTOCOL)

        self.read_debug()

    def load_image_sequences(self):
        img_folders = []
        if self.include_all:  # include all the folders in one directory -- for duke
            img_folders = os.listdir(self.img_dir)
        elif self.extend:
            img_folders = [str(k) for k in range(101, 1040)]

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
    from utils.image import seq_show
    from utils.data import get_path

    np.set_printoptions(precision=4)

    # unlabelset = FolderUnlabelDataset(img_dir='/datadrive/person/dirimg',seq_length = 24, extend=True, data_aug=True)#,data_file='duke_unlabeldata.pkl')
    # unlabelset = FolderUnlabelDataset(seq_length=24, data_aug=True, extend=True, data_file='drone_ucf_unlabeldata.pkl')
    img_dir = 'DukeMCMT/heading'
    unlabelset = FolderUnlabelDataset(
        img_dir=get_path(img_dir), seq_length=24, data_aug=True, include_all=True)

    dataloader = DataLoader(unlabelset)
    count = 20
    for sample in dataloader:
        seq_show(sample.squeeze().numpy(), scale=0.8)

        count -= 1
        if count < 0:
            break

if __name__ == '__main__':
    main()
