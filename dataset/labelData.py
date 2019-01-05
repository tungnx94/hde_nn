# Combine two labeled dataset together
import sys
sys.path.insert(0, "..")

from utils import get_path

from generalData import GeneralDataset
from trackingLabelData import TrackingLabelDataset
from folderLabelData import FolderLabelDataset


PersonFiles = 'combined_data2/train/annotations/person_annotations.csv'
DukeLabelFile = 'DukeMCMT/trainval_duke.txt'
HandLabelFolder = 'label'


Factors = [2, 1, 20]


class LabelDataset(GeneralDataset):

    def init_datasets(self):
        if self.factors is None:
            self.factors = Factors

        virat = TrackingLabelDataset("combine-train",
            data_file=get_path(PersonFiles), data_aug=True, mean=self.mean, std=self.std)  

        duke = TrackingLabelDataset("duke-train",
            data_file=get_path(DukeLabelFile), data_aug=True, mean=self.mean, std=self.std)

        handlabel = FolderLabelDataset("handlabel-train",
            img_dir=get_path(HandLabelFolder), data_aug=True, mean=self.mean, std=self.std)

        self.datasets = [virat, duke, handlabel]


def main():
    # test
    import cv2
    import numpy as np
    from utils import seq_show
    from generalData import DataLoader

    np.set_printoptions(precision=4)

    labeldataset = LabelDataset(balance=True)
    dataloader = DataLoader(labeldataset, batch_size=16)

    # import ipdb;ipdb.set_trace()
    print len(labeldataset)

    count = 10
    for sample in dataloader:
        print sample['label'], sample['img'].size()
        seq_show(sample['img'].numpy(),
                 dir_seq=sample['label'].numpy(), scale=0.5)

        count -= 1
        if count < 0:
            break

if __name__ == '__main__':
    main()
