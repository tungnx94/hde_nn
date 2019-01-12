
import sys
sys.path.insert(0, "..")

import os
import cv2
import numpy as np
import pandas as pd

from generalData import SingleDataset


class SequenceDataset(SingleDataset):

    def __init__(self, name, img_size, data_aug, maxscale, mean, std, seq_length):
        SingleDataset.__init__(self, name, img_size, data_aug, maxscale, mean, std)
        self.seq_length = seq_length

    def read_debug(self):
        print '{}: {} sequences, {} images'.format(self, self.len(), self.len() * self.seq_length)

    def save_sequences(self, seq):
        # add new sequence to list if long enough
        if len(seq) >= self.seq_length:
            for start_t in range(len(seq) - self.seg_length + 1)
                self.items.append(seq[start_t: start_t + self.seq_length])


class SequenceUnlabelData(SequenceDataset):
    
    def __getitem__(self, idx):
        flip = self.get_flipping()

        out_seq = []
        for img_path in self.items[idx]:
            img = cv2.imread(img_path)
            out_img = self.augment_image(img, flip)
            out_seq.append(out_img)

        return np.array(out_seq)

class SequenceLabelData(SequenceDataset):
    def __getitem__(self, idx):

        flip = self.get_flipping()

        out_seq = []
        for img_path in self.items[idx]:
            img = cv2.imread(img_path)
            out_img = self.augment_image(img, flip)
            out_seq.append(out_img)

        return np.array(out_seq)

        

class SingleSequenceDataset(SequenceDataset):
    # extended by DukeSequenceDataset, ViratSequenceDataset

    def __init__(self, name, img_size, data_aug, maxscale, mean, std, seq_length):
        super(SingleSequenceDataset, self).__init__(
            name, img_size, data_aug, maxscale, mean, std, seq_length)

    def __getitem__(self, idx):
        ep_idx, idx = self.get_indexes(idx)

        # random fliping
        flipping = self.get_flipping()

        imgseq = []
        labelseq = []
        for k in range(self.seq_length):
            img = cv2.imread(self.img_seqs[ep_idx][idx + k][0])

            angle = self.img_seqs[ep_idx][idx + k][1]
            label = label_from_angle(angle)

            out_img, label = self.get_img_and_label(img, label, flipping)

            imgseq.append(out_img)
            labelseq.append(label)

        return {'imgseq': np.array(imgseq), 'labelseq': np.array(labelseq)}
