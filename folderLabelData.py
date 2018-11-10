import os
import random
import numpy as np

from generalData import SingleDataset
from utils import im_scale_norm_pad, img_denormalize, im_crop, im_hsv_augmentation


class FolderLabelDataset(SingleDataset):

    def __init__(self, img_dir='/home/wenshan/headingdata/label',
                 img_size=192, data_aug=False, maxscale=0.1, mean=[0, 0, 0], std=[1, 1, 1]):

        super(FolderLabelDataset, self)__init__(img_size, data_aug, maxscale, mean, std)

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
    import cv2
    from torch.utils.data import DataLoader
    from utils import seq_show, put_arrow

    np.set_printoptions(precision=4)

    facingDroneLabelDataset = FolderLabelDataset(
        img_dir='/datadrive/3DPES/facing_labeled', data_aug=True)

    dataloader = DataLoader(facingDroneLabelDataset,
                            batch_size=4, shuffle=True, num_workers=1)

    # import ipdb; ipdb.set_trace()
    for sample in dataloader:
        print sample['label'], sample['img'].size()
        seq_show(sample['img'].numpy(), scale=0.3)

    """
    for k in range(100):
        sample = facingDroneLabelDataset[k * 100]
        img = sample['img']
        label = sample['label']
        print img.dtype, label
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        img = img_denormalize(img)
        img = put_arrow(img, label)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    """

if __name__ == '__main__':
    main()
