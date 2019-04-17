import sys
sys.path.insert(0, "..")

import random
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset
import utils


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False):
        super(DataLoader, self).__init__(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        self.epoch = 0
        self.dataset = dataset
        self.data_iter = iter(self)

        self.max_iter_count = len(dataset) // batch_size

    def next_sample(self):
        """ get next batch, update data_iter and epoch if needed """
        try:
            sample = self.data_iter.next()
        except:
            self.reset_iteration()
            self.epoch += 1
            sample = self.data_iter.next()

        return sample

    def reset_iteration(self):
        self.dataset.shuffle()
        self.data_iter = iter(self)

    def max_iteration(self):
        return self.max_iter_count

class GeneralDataset(Dataset):

    def __init__(self, config, auto_shuffle=False):
        Dataset.__init__(self)
        self.name = config["name"]
        self.auto_shuffle = auto_shuffle
        self.items = []

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.items)

    def resize(self, nsize):
        self.reorder()
        if isinstance(nsize, float):
            nsize = (int)(nsize * len(self))

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
        file = open(fname, 'rb')
        self.items = pickle.load(file)

    def read_debug(self):
        print("{}: {} samples".format(self, len(self)))


class SingleDataset(GeneralDataset):

    def __init__(self, config, mean=[0, 0, 0], std=[1, 1, 1], img_size=192, maxscale=0.1, auto_shuffle=False):

        GeneralDataset.__init__(self, config, auto_shuffle)
        self.path = utils.get_path(config["path"])

        self.aug = config["aug"]
        self.img_size = img_size
        self.maxscale = maxscale

        self.mean = np.array(mean)
        self.std = np.array(std)

        if "saved" in config:
            self.load(config["saved"])
        else:
            self.init_data()

        self.read_debug()

    def init_data(self):
        pass

    def get_flipping(self):  # random fliping with probability 0.5
        return (self.aug and random.random() > 0.5)

    def augment_image(self, img, flipping):
        # augment image to make "new" data
        if self.aug:
            img = utils.im_hsv_augmentation(img)
            img = utils.im_crop(img, maxscale=self.maxscale)

        out_img = utils.im_scale_norm_pad(img, self.mean, self.std, out_size=self.img_size,
                                    # down_reso=True,
                                    flip=flipping)

        return out_img

    def augment_label(self, label, flipping):
        if self.aug and flipping:
            return np.array([label[0], -label[1]])
        else:
            return label

    def augment_direction(self, direction, flipping):
        if self.aug and flipping:
            return FlipDir[direction]
        else:
            return direction

    def calculate_mean_std(self):
        return utils.calculate_mean_std(self.image_paths())

    def image_paths(self):
        return []


class MixDataset(GeneralDataset):

    def __init__(self, config, saved_file=None, auto_shuffle=False):
        GeneralDataset.__init__(self, config, auto_shuffle)

        if "saved" in config:
            self.load(config["saved"])

    def add(self, dataset, factor=1):
        for idx in range(len(dataset)):
            for i in range(factor):
                self.items.append({"dataset": dataset, "idx": idx})

        self.reorder()

    def __getitem__(self, idx):
        dataset = self.items[idx]["dataset"]
        idx = self.items[idx]["idx"]

        return dataset[idx]
