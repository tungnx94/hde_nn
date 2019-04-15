# wrapper for unlabeled data
import os
import cv2 
import numpy as np
import pandas as pd

from os.path import join
from .generalData import SingleDataset

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

    def image_paths(self):
        return self.items