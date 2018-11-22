# Combine two labeled dataset together
import sys
sys.path.insert(0, "..")

from utils.data import get_path
from generalData import GeneralDataset
from folderUnlabelData import FolderUnlabelDataset

Base = "/home/mohammad/projects/facing_icra/data"
UcfFile = 'ucf_unlabeldata.pkl'
DukeFile = 'duke_unlabeldata.pkl'

UcfPath = get_path(UcfFile, base_folder=Base)
DukePath = get_path(DukeFile, base_folder=Base)

Factors = [4, 1]

class UnlabelDataset(GeneralDataset):

    def __init__(self, seq_length, balance=False, mean=[0, 0, 0], std=[1, 1, 1]):
        self.seg_length = seq_length

        super(LabelDataset, self).__init__(balance, mean, std)

    def init_datasets(self):
        if self.factors is None
            self.factors = Factors

        ucf = FolderUnlabelDataset(
            seq_length=self.seq_length, data_aug=True, data_file=UcfPath, mean=self.mean, std=self.std)  # 940
        duke = FolderUnlabelDataset(
            seq_length=self.seq_length, data_aug=True, data_file=DukePath, mean=self.mean, std=self.std)  # 3997

        self.datasets = [ucf, duke]


def main():
    # test
    import cv2
    import numpy as np
    from utils.image import seq_show, put_arrow
    from generalData import DataLoader

    np.set_printoptions(precision=4)

    unlabeldataset = UnlabelDataset(seq_length=24, balance=True)
    dataloader = DataLoader(unlabeldataset, batch_size=1,
                            shuffle=True, num_workers=1)

    # import ipdb;ipdb.set_trace()
    print len(unlabeldataset)
    for sample in dataloader:
        seq_show(sample.squeeze().numpy())

    """
    # datalist=[0,69679,69680,69680*2-1,69680*2,364785,364786]
    for k in dataloader:
        sample = labeldataset[k]
        img = sample['img']
        label = sample['label']
        print img.dtype, label
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        img = img_denormalize(img)
        img = put_arrow(img, label)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    """

if __name__ == '__main__':
    main()
