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


if __name__ == '__main__':
    from utils import seq_show
    from generalData import DataLoader

    dataset = UnlabelDataset(seq_length=24, balance=True)
    dataloader = DataLoader(dataset)
    count = 10
    
    for count in range(10):
        sample = dataloader.next_sample()
        seq_show(sample.squeeze().numpy())
