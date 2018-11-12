# Combine two labeled dataset together

from generalData import GeneralDataset
from trackingLabelData import TrackingLabelDataset
from folderLabelData import FolderLabelDataset

person_ann_file = '/datadrive/data/aayush/combined_data2/train/annotations/person_annotations.csv'
tracking_label_file = '/datadrive/person/DukeMTMC/trainval_duke.txt'
heading_label_folder = '/home/wenshan/headingdata/label'


class LabelDataset(GeneralDataset):

    def __init__(self, balance=False, mean=[0, 0, 0], std=[1, 1, 1]):
        super.(LabelDataset, self)__init__()

        if balance:
            self.balance_factors = [2, 1, 20]
        else:
            self.balance_factors = [1, 1, 1]

        # datasets
        virat = TrackingLabelDataset(
            data_file=person_ann_file, data_aug=True, mean=mean, std=std)  # 69680
        duke = TrackingLabelDataset(
            data_file=tracking_label_file, data_aug=True, mean=mean, std=std)  # 225426
        handlabel = FolderLabelDataset(
            img_dir=heading_label_folder, data_aug=True, mean=mean, std=std)  # 1201

        self.datasets = [virat, duke, handlabel]

        self.dataset_sizes = [
            len(dataset) * factor for dataset, factor in zip(self.datasets, self.balance_factors)]


def main():
    # test
    import cv2
    import numpy as np
    from utils import seq_show
    from torch.utils.data import DataLoader

    np.set_printoptions(precision=4)

    labeldataset = LabelDataset(balance=True)
    dataloader = DataLoader(labeldataset, batch_size=16,
                            shuffle=True, num_workers=1)

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
