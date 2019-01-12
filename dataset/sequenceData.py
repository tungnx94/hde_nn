import cv2
import numpy as np
import pandas as pd

from generalData import SingleDataset


class SequenceDataset(SingleDataset):

    def __init__(self, name, img_size, data_aug, maxscale, mean, std, seq_length, saved_file=None):
        self.seq_length = seq_length

        SingleDataset.__init__(self, name, img_size, data_aug, maxscale, mean, std)

    def read_debug(self):
        print '{}: {} sequences, {} images'.format(self, len(self), len(self) * self.seq_length)

    def save_sequence(self, seq):
        # add new sequence to list if long enough
        if len(seq) >= self.seq_length:
            for start_t in range(len(seq) - self.seq_length + 1):
                self.items.append(seq[start_t: start_t + self.seq_length])


class SequenceUnlabelDataset(SequenceDataset):

    def __getitem__(self, idx):
        flip = self.get_flipping()

        out_seq = []
        for img_path in self.items[idx]:
            img = cv2.imread(img_path)
            out_img = self.augment_image(img, flip)
            out_seq.append(out_img)

        return np.array(out_seq)


class SequenceLabelDataset(SequenceDataset):

    def __getitem__(self, idx):
        flip = self.get_flipping()

        out_seq = []
        label_seq = []
        for img_path, label in self.items[idx]:
            img = cv2.imread(img_path)
            out_img = self.augment_image(img, flip)
            out_label = self.augment_label(label, flip)

            out_seq.append(out_img)
            label_seq.append(out_label)

        return (np.array(out_seq), np.array(label_seq))
