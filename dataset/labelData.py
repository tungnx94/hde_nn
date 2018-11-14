# Combine two labeled dataset together

from generalData import GeneralDataset
from trackingLabelData import TrackingLabelDataset
from folderLabelData import FolderLabelDataset

PersonFiles = '../data/combined_data2/train/annotations/person_annotations.csv'
DukeLabelFile = '../data/person/DukeMTMC/trainval_duke.txt'
HandLabelFolder = '../data/headingdata/label'

Factors = [2, 1, 20]

class LabelDataset(GeneralDataset):

    def init_datasets(self):
        if self.factors is None:
            self.factors = Factors

        virat = TrackingLabelDataset(
            data_file=PersonFiles, data_aug=True, mean=self.mean, std=self.std)  # 69680
        duke = TrackingLabelDataset(
            data_file=DukeLabelFile, data_aug=True, mean=self.mean, std=self.std)  # 225426
        handlabel = FolderLabelDataset(
            img_dir=HandLabelFolder, data_aug=True, mean=self.mean, std=self.std)  # 1201

        self.datasets = [virat, duke, handlabel]


def main():
    # test
    import cv2
    import numpy as np
    from ..utils.image import seq_show
    from generalData import DataLoader

    np.set_printoptions(precision=4)

    labeldataset = LabelDataset(balance=True)
    dataloader = DataLoader(labeldataset, batch_size=16)

    # import ipdb;ipdb.set_trace()
    print len(labeldataset)

    for sample in dataloader:
        print sample['label'], sample['img'].size()
        seq_show(sample['img'].numpy(),
                 dir_seq=sample['label'].numpy(), scale=0.5)

    """ old code 
    # datalist=[0,69679,69680,69680*2-1,69680*2,364785,364786] ?
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
