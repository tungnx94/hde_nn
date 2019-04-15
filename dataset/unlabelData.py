# wrapper for unlabeled data
import sys
sys.path.insert(0, "..")

import os
import cv2 
import numpy as np
import pandas as pd

from os.path import join
from .generalData import SingleDataset

import utils


class UnlabelDataset(SingleDataset):

    def init_data(self):
        img_dir = os.path.dirname(self.path)
        data = pd.read_csv(self.path).to_dict(orient='list')
        img_folders = data["folder"]

        for folder in img_folders:
            folder_path = join(img_dir, folder)

            img_list = [file for file in os.listdir(
                folder_path) if file.endswith('.jpg') or file.endswith('png')]
            seq = [join(folder_path, img) for img in sorted(img_list)] # absolute path

            self.items.extend(seq)

    def __getitem__(self, idx):
        img = cv2.imread(self.items[idx])
        return self.augment_image(img, self.get_flipping())

    def calculate_mean_std(self):
        img_paths = self.items
        count = len(img_paths) * 192 * 192

        # calculate mean  
        mean = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            im_mean = np.mean(img, axis=(0, 1))
            mean.append(im_mean)

        mean = np.mean(np.array(mean), axis=0)

        # calculate std 
        #std = np.zeros(3)
        std = []
        for img_path in img_paths:
            img = cv2.imread(img_path)

            sqr_diff = (img - mean) ** 2
            std.append(np.sum(sqr_diff, axis=(0, 1)))

        std = np.sqrt(np.sum(std, axis=0) / (count-1)) 

        return (mean, std)