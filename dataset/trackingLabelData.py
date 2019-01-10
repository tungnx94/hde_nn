# Wrapper for Duke & VIRAT single image labeled datasets

import sys
sys.path.insert(0, "..")

import cv2
import random
import os.path

import numpy as np
import pandas as pd
import config as cnf

from utils import label_from_angle
from generalData import SingleDataset


class TrackingLabelDataset(SingleDataset):

    def __init__(self, name, data_file, img_size=192, data_aug=False, maxscale=0.1,
                 mean=[0, 0, 0], std=[1, 1, 1]):

        super(TrackingLabelDataset, self).__init__(name, img_size, data_aug, maxscale, mean, std)

        self.data_file = data_file # absolute path needed

        # save image paths & directions
        self.items = []
        if data_file.endswith(".csv"):  # VIRAT
            self.items = pd.read_csv(data_file)
        else:
            raise Exception

        self.N = len(self.items)
        self.read_debug()

    def __getitem__(self, idx):
        point_info = self.items.iloc[idx]
        # read image
        img_name = point_info['path']
        img = cv2.imread(img_name)

        if img is None:
            print 'error reading image:', img_name
            return

        # get label
        label = label_from_angle(point_info['direction_angle'])

        flipping = self.get_flipping()
        out_img, label = self.get_img_and_label(img, label, flipping)

        return {'img': out_img, 'label': label}


if __name__ == '__main__':
    from utils import get_path, seq_show
    from generalData import DataLoader
    
    duke = TrackingLabelDataset("duke",
        data_file=get_path('DukeMTMC/train/person.csv'), data_aug=True)

    virat = duke = TrackingLabelDataset("virat",
        data_file=get_path('VIRAT/train/person.csv'), data_aug=True)

    for dataset in [duke, virat]:
        dataloader = DataLoader(dataset, batch_size=16)

        for count in range(5):
            sample = dataloader.next_sample()

            print sample['label'], sample['img'].size()
            seq_show(sample['img'].numpy(),
                dir_seq=sample['label'].numpy(), scale=0.5)
