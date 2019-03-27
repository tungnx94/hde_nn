import sys
sys.path.insert(0, '..')

import cv2
import numpy as np
import pandas as pd

from generalData import SingleDataset
from utils import one_hot

class SequenceDataset(SingleDataset):

    def __init__(self, name, path=None, img_size=192, seq_length=12, 
                data_aug=False, maxscale=0.1, mean=[0, 0, 0], std=[1, 1, 1],
                saved_file=None, auto_shuffle=False, testing=False):

        self.seq_length = seq_length
        SingleDataset.__init__(self, name, path, img_size, data_aug, maxscale, mean, std, saved_file, auto_shuffle, testing)

    def read_debug(self):
        print '{}: {} sequences, {} images'.format(self, len(self), len(self) * self.seq_length)

    def save_sequence(self, seq):
        ### add new sequence to list if long enough ###
        """
        # Approach 1: greedy adding -> resulting in very similar sequences
        if len(seq) >= self.seq_length:
            for start_t in range(len(seq) - self.seq_length + 1):
                self.items.append(seq[start_t: start_t + self.seq_length])
        """

        # Approach 2: adding in segment
        start = 0
        while start + self.seq_length < len(seq):
            self.items.append(seq[start: start + self.seq_length])
            start += self.seq_length


class SequenceLabelDataset(SequenceDataset):
    
    def __getitem__(self, idx):
        flip = self.get_flipping()

        out_seq = []
        label_seq = []
        dir_seq = []

        for sample in self.items[idx]:
            img_path = sample[0]
            label = sample[1]
            direction = sample[2] # direction

            img = cv2.imread(img_path)
            out_img = self.augment_image(img, flip)
            out_label = self.augment_label(label, flip)
            # out_direction = one_hot(self.augment_direction(direction, flip))

            out_seq.append(out_img)
            label_seq.append(out_label)

        return (np.array(out_seq), np.array(label_seq))
