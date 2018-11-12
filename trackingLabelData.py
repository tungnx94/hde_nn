import random
import os.path

import numpy as np
import pandas as pd
import config as cnf

from utils import label_from_angle
from generalData import SingleDataset

default_data_file = '/datadrive/data/aayush/combined_data2/train/annotations/person_annotations.csv'


class TrackingLabelDataset(SingleDataset):

    def __init__(self, data_file, img_size=192, data_aug=False, maxscale=0.1,
                 mean=[0, 0, 0], std=[1, 1, 1]):
        super(TrackingLabelDataset, self)__init__(img_size, data_aug, maxscale, mean, std)

        self.data_file = data_file

        # save image paths & directions
        self.items = []
        if data_file.endswith(".csv"):  # .csv file
            self.items = pd.read_csv(data_file)

        else:  # text file used by DukeMTMC dataset

            img_dir = os.path.dirname(data_file)
            # img_dir = join(img_dir,'heading') # a subdirectory

            with open(data_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                img_name, angle = line.strip().split(' ')
                self.items.append(
                    {'path': os.path.join(img_dir, img_name), 'direction_angle': angle})

        self.N = len(self.items)

        # debug
        print 'Read #images: ', len(self.items)

    def __getitem__(self, idx):
        if self.data_file.endswith('csv'):
            point_info = self.items.iloc[idx]
        else:
            point_info = self.items[idx]
        # print(point_info)

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


def main():
    # test
    import cv2
    from utils import seq_show, put_arrow
    from torch.utils.data import DataLoader

    np.set_printoptions(precision=4)

    # trackingLabelDataset = TrackingLabelDataset(data_aug = True, maxscale=0.1)
    # trackingLabelDataset = TrackingLabelDataset(data_file='/datadrive/data/aayush/combined_data2/train/annotations/car_annotations.csv')
    trackingLabelDataset = TrackingLabelDataset(
        data_file=default_data_file, data_aug=True)

    dataloader = DataLoader(trackingLabelDataset,
                            batch_size=16, shuffle=True, num_workers=1)

    # import ipdb;ipdb.set_trace()
    for sample in dataloader:
        print sample['label'], sample['img'].size()
        seq_show(sample['img'].numpy(),
                 dir_seq=sample['label'].numpy(), scale=0.5)

    """
    print len(trackingLabelDataset)
    for k in range(1000):
        img = trackingLabelDataset[k*10]['img']
        label = trackingLabelDataset[k*10]['label']
        print img.dtype, label
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        img = img_denormalize(img)
        img = put_arrow(img, label)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    """

if __name__ == '__main__':
