import sys
sys.path.insert(0, '..')
from utils import get_path

from generalData import DataLoader, MixDataset
from singleLabelData import SingleLabelDataset 
from folderUnlabelData import FolderUnlabelDataset
from dukeSeqLabelData import DukeSeqLabelDataset
from viratSeqLabelData import ViratSeqLabelDataset

class DatasetLoader(object):

    def __init__(self, mean, std):
        self.name = "Dataset-Loader"
        self.mean = mean
        self.std = std

    def set_mean(self, mean):
        self.mean = mean

    def set_std(self, std):
        self.std = std

    def loader(self, dataset, batch_size, shuffle=True, num_workers=4):
        return DataLoader(dataset, batch_size, shuffle, num_workers)

    def folder_unlabel(self, name, path, data_aug=True):
        return FolderUnlabelDataset(name, path=get_path(path), data_aug=data_aug, mean=self.mean, std=self.std)

    def single_label(self, name, path, data_aug=True):
        return SingleLabelDataset(name, path=get_path(path), data_aug=data_aug, mean=self.mean, std=self.std)

    def duke_seq(self, name, path, seq_length, data_aug=True):
        return DukeSeqLabelDataset(name, path=get_path(path), seq_length=seq_length, data_aug=data_aug, mean=self.mean, std=self.std)

    def virat_seq(self, name, path, seq_length, data_aug=True):
        return ViratSeqLabelDataset(name, path=get_path(path), seq_length=seq_length, data_aug=data_aug, mean=self.mean, std=self.std)

    def mix(self, name, sets, factors=None):
        mixset = MixDataset(name)

        if factors == None:
            factors = [1] * len(sets)

        for dts, factor in zip(sets, factors):
            mixset.add(dts, factor) 

        return mixset
