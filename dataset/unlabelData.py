# Combine two labeled dataset together
import sys
sys.path.insert(0, "..")

from utils import get_path
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
        self.seq_length = seq_length

        super(UnlabelDataset, self).__init__(balance, mean, std)

    def init_datasets(self):
        if self.factors is None:
            self.factors = Factors

        ucf = FolderUnlabelDataset("ucf-unlabel-train",
            seq_length=self.seq_length, data_aug=True, data_file=UcfPath, mean=self.mean, std=self.std)  # 940
        
        duke = FolderUnlabelDataset("duke-unlabel-train",
            seq_length=self.seq_length, data_aug=True, data_file=DukePath, mean=self.mean, std=self.std)  # 3997

        self.datasets = [ucf, duke]

def main():
    # test
    import cv2
    import numpy as np
    from utils import seq_show, put_arrow
    from generalData import DataLoader

    np.set_printoptions(precision=4)

    unlabeldataset = UnlabelDataset(seq_length=24, balance=True)
    dataloader = DataLoader(unlabeldataset, batch_size=1,
                            shuffle=True, num_workers=1)
    count = 10
    
    for sample in dataloader:
        seq_show(sample.squeeze().numpy())
        count -= 10
        if count < 0:
            break

if __name__ == '__main__':
    main()
