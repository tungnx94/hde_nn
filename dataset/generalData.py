import sys
sys.path.insert(0, "..")

import random
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset
from utils import im_scale_norm_pad, im_crop, im_hsv_augmentation


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=1):
        super(DataLoader, self).__init__(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)
        self.epoch = 0
        self.data_iter = iter(self)

    def next_sample(self):
        """ get next batch, update data_iter and epoch if needed """
        try:
            sample = self.data_iter.next()
        except:
            self.epoch += 1
            self.data_iter = iter(self)
            sample = self.data_iter.next()

        return sample


class GeneralDataset(Dataset):

    def __init__(self, name, auto_shuffle=False):
        Dataset.__init__(self)
        self.name = name
        self.auto_shuffle = auto_shuffle
        self.items = []

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.items)

    def resize(self, nsize):
        if nsize <= len(self.items):
            self.items = self.items[:nsize]

        self.reorder()

    def shuffle(self):
        random.shuffle(self.items)

    def reorder(self):
        if self.auto_shuffle:
            self.shuffle()

    def save(self, fname):
        file = open(fname, "wb")
        pickle.dump(self.items, file, pickle.HIGHEST_PROTOCOL)

    def load(self, fname):
        file = open(fname, "rb")
        self.items = pickle.load(file)

    def read_debug(self):
        print "{}: {} samples".format(self, len(self))


class SingleDataset(GeneralDataset):

    def __init__(self, name, img_size, data_aug, maxscale, mean, std, saved_file=None, auto_shuffle=False):
        self.aug = data_aug
        self.mean = mean
        self.std = std
        self.maxscale = maxscale
        self.img_size = img_size

        GeneralDataset.__init__(self, name, auto_shuffle)

        if saved_file is not None:
            self.load(saved_file)
        else:
            self.init_data()

        self.read_debug()

    def init_data(self):
        pass

    def get_flipping(self):  # random fliping with probability 0.5
        return (self.aug and random.random() > 0.5)

    def augment_image(self, img, flipping):
        # to make "new" data
        if self.aug:
            img = im_hsv_augmentation(img)
            img = im_crop(img, maxscale=self.maxscale)

        out_img = im_scale_norm_pad(
            img, out_size=self.img_size, mean=self.mean, std=self.std, down_reso=True, flip=flipping)

        return out_img

    def augment_label(self, label, flipping):
        if self.aug and flipping:
            return np.array([label[0], -label[1]])
        else:
            return label


class MixDataset(GeneralDataset):

    def __init__(self, name, saved_file=None, auto_shuffle=False):
        GeneralDataset.__init__(self, name, auto_shuffle)

        if saved_file is not None:
            self.load(saved_file)

    def add(self, dataset, factor=1):
        for idx in range(len(dataset)):
            for i in range(factor):
                self.items.append({"dataset": dataset, "idx": idx})

        self.reorder()

    def __getitem__(self, idx):
        dataset = self.items[idx]["dataset"]
        idx = self.items[idx]["idx"]

        return dataset[idx]
