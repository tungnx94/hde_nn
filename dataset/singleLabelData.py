# Wrapper for Duke & VIRAT single image labeled datasets
import sys
sys.path.insert(0, "..")

import os
import cv2
import numpy as np
import pandas as pd

from generalData import SingleDataset

# TODO: fix applied mean & std in real training data


class SingleLabelDataset(SingleDataset):

    def __init__(self, name, data_file, img_size=192, data_aug=False, maxscale=0.1,
                 mean=[0, 0, 0], std=[1, 1, 1]):

        super(SingleLabelDataset, self).__init__(
            name, img_size, data_aug, maxscale, mean, std)
        self.data_file = data_file  # absolute path

        # save image paths & directions
        if data_file.endswith(".csv"):  # VIRAT
            data = pd.read_csv(data_file)
        else:
            raise Exception

        base_folder = os.path.dirname(data_file)
        # each element is (image, label)
        for point in data:
            img_path = os.path.join(base_folder, point['path'])
            label = np.array([point['sin'], point['cos']], dtype=np.float32)
            self.items.append((img_path, label))

        self.read_debug()

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = cv2.imread(img_path)
        flip = self.get_flipping()

        out_img = self.augment_image(img, flip)
        out_label = self.augment_label(label, flip)

        return out_img, out_label


if __name__ == '__main__':
    from utils import get_path, seq_show
    from generalData import DataLoader

    duke = SingleLabelDataset("duke",
                              data_file=get_path('DukeMTMC/train/person.csv'), data_aug=True)

    virat = SingleLabelDataset("virat",
                               data_file=get_path('VIRAT/train/person.csv'), data_aug=True)

    3dpes = SingleLabelDataset("3dpes",
                               data_file=get_path('3DPES/person.csv'), data_aug=False)

    for dataset in [duke, virat, 3dpes]:
        print dataset

        dataloader = DataLoader(dataset, batch_size=8)
        for count in range(3):
            img, label = dataloader.next_sample()

            seq_show(img.numpy(),
                     dir_seq=label.numpy(), scale=0.5)
