import sys
sys.path.insert(0, "..")

import os
import cv2
import random
import numpy as np

from generalData import SingleDataset
from utils im_scale_norm_pad, img_denormalize, im_crop, im_hsv_augmentation


class FolderLabelDataset(SingleDataset):

    def __init__(self, img_dir,
                 img_size=192, data_aug=False, maxscale=0.1, mean=[0, 0, 0], std=[1, 1, 1]):

        super(FolderLabelDataset, self).__init__(img_size, data_aug, maxscale, mean, std)

        # self.dir2ind = {'n': 0,'ne': 1,'e': 2, 'se': 3,'s': 4,'sw': 5,'w': 6,'nw': 7}
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

        print 'Read #images: ', self.__len__()

    def __getitem__(self, idx):
        img = cv2.imread(self.img_names[idx])  # in bgr
        label = np.array(self.labels[idx], dtype=np.float32)

        flipping = self.get_flipping()
        out_img, label = self.get_img_and_label(img, label, flipping)

        return {'img': out_img, 'label': label}


def main():
    # test
    from generalData import DataLoader
    from utils import get_path, seq_show, put_arrow

    np.set_printoptions(precision=4)

    test_img_dir = '3DPES/facing_labeled'
    facingDroneLabelDataset = FolderLabelDataset(
        img_dir=get_path(test_img_dir), data_aug=True)

    dataloader = DataLoader(facingDroneLabelDataset,
                            batch_size=4, shuffle=True, num_workers=1)

    # import ipdb; ipdb.set_trace()
    count = 20
    for sample in dataloader:
        print sample['label'], sample['img'].size()
        seq_show(sample['img'].numpy(), scale=0.3)
        
        count -= 1
        if count < 0:
            break

if __name__ == '__main__':
    main()
