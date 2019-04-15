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

            #self.save_sequence(seq)




    def __getitem__(self, idx):
        flip = self.get_flipping()

        out_seq = []
        for img_path in self.items[idx]:
            img = cv2.imread(img_path)
            out_img = self.augment_image(img, flip)
            out_seq.append(out_img)

        return np.array(out_seq)
