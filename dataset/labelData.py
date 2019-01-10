# Combine two labeled dataset together
import sys
sys.path.insert(0, "..")

from utils import get_path

from generalData import GeneralDataset
from trackingLabelData import TrackingLabelDataset
from folderLabelData import FolderLabelDataset


PersonFiles = 'VIRAT/train/annotations/person_annotations.csv'
DukeLabelFile = 'DukeMTMC/trainval_duke.txt'
HandLabelFolder = 'label'


Factors = [2, 1, 20]


class LabelDataset(GeneralDataset):

    def init_datasets(self):
        if self.factors is None:
            self.factors = Factors

        virat = TrackingLabelDataset("VIRAT-train",
            data_file=get_path(PersonFiles), data_aug=True, mean=self.mean, std=self.std)  

        duke = TrackingLabelDataset("DUKE-train",
            data_file=get_path(DukeLabelFile), data_aug=True, mean=self.mean, std=self.std)

        handlabel = FolderLabelDataset("handlabel-train",
            img_dir=get_path(HandLabelFolder), data_aug=True, mean=self.mean, std=self.std)

        self.datasets = [virat, duke, handlabel]


if __name__ == '__main__':
    from utils import seq_show
    from generalData import DataLoader

    labeldataset = LabelDataset(balance=True)
    dataloader = DataLoader(labeldataset, batch_size=16)

    for count in range(10):
        sample = dataloader.next_sample()

        print sample['label'], sample['img'].size()
        seq_show(sample['img'].numpy(),
                 dir_seq=sample['label'].numpy(), scale=0.5)
