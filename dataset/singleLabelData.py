# Wrapper for Duke & VIRAT single image labeled datasets
import os
import cv2
import numpy as np
import pandas as pd

from .generalData import SingleDataset


class SingleLabelDataset(SingleDataset):

    def init_data(self):
        data_file = self.path 
        data = pd.read_csv(data_file).to_dict(orient='records')
        base_folder = os.path.dirname(data_file)

        # each element is (image, label)
        for point in data:
            img_path = os.path.join(base_folder, point['path'])
            angle = point['angle']
            label = np.array(
                [np.sin(angle), np.cos(angle)], dtype=np.float32)
            group = point['direction']

            self.items.append((img_path, label, group))

    def __getitem__(self, idx):
        img_path, label, gr = self.items[idx]
        img = cv2.imread(img_path)
        flip = self.get_flipping()

        out_img = self.augment_image(img, flip)
        out_label = self.augment_label(label, flip)
    
        return (out_img, out_label)
