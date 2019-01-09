import sys
sys.path.insert(0, "..")

import os
import cv2
import random
import numpy as np

from generalData import SingleDataset


class FolderLabelDataset(SingleDataset):

    def __init__(self, name, img_dir,
                 img_size=192, data_aug=False, maxscale=0.1, mean=[0, 0, 0], std=[1, 1, 1]):

        SingleDataset.__init__(self, name, img_size, data_aug, maxscale, mean, std)
        self.dir2val = {'n':  [1., 0.],
                        'ne': [0.707, 0.707],
                        'e':  [0., 1.],
                        'se': [-0.707, 0.707],
                        's':  [-1., 0.],
                        'sw': [-0.707, -0.707],
                        'w':  [0., -1.],
                        'nw': [0.707, -0.707]}

        self.img_names = []
        self.labels = []

        for cls_folder in os.listdir(img_dir):

            cls_val = self.dir2val[cls_folder]

            path = os.path.join(img_dir, cls_folder)

            for img_name in os.listdir(path):
                if img_name.endswith(".jpg") or img_name.endswith(".png"):
                    self.img_names.append(os.path.join(path, img_name))
                    self.labels.append(cls_val)

        self.N = len(self.img_names)

        self.read_debug()

    def __getitem__(self, idx):
        img = cv2.imread(self.img_names[idx])  # in bgr
        label = np.array(self.labels[idx], dtype=np.float32)

        flipping = self.get_flipping()
        out_img, label = self.get_img_and_label(img, label, flipping)

        return {'img': out_img, 'label': label}


if __name__ == '__main__': # test
    from generalData import DataLoader
    from utils import get_path, seq_show

    dataset = FolderLabelDataset("3DPES",
        img_dir=get_path("3DPES"), data_aug=True)
    dataloader = DataLoader(dataset,
                            batch_size=4, shuffle=True, num_workers=1)

    for count in range(20):
        sample = dataloader.next_sample()

        print sample['label'], sample['img'].size()
        seq_show(sample['img'].numpy(), scale=0.3)