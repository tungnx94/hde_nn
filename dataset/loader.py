from .generalData import DataLoader, MixDataset
from .singleLabelData import SingleLabelDataset 
from .sequenceUnlabelData import SequenceUnlabelDataset
from .dukeSeqLabelData import DukeSeqLabelDataset
from .viratSeqLabelData import ViratSeqLabelDataset

"""
Dataset type
0: single 
1: sequence unlabel
2: duke seq
3: virat seq
"""

class DatasetLoader(object):

    def __init__(self, mean, std):
        self.name = "Dataset-Loader"
        self.mean = mean
        self.std = std

    def set_mean(self, mean):
        self.mean = mean

    def set_std(self, std):
        self.std = std

    def loader(self, dataset, batch_size=1, shuffle=True, num_workers=4):
        return DataLoader(dataset, batch_size, shuffle, num_workers)        

    def mix(self, name, sets, factors=None):
        # currently unused
        mixset = MixDataset(name)

        if factors == None:
            factors = [1] * len(sets)

        for dts, factor in zip(sets, factors):
            mixset.add(dts, factor) 

        return mixset

    def load(self, config):
        t = config["type"]
        if t == 0:
            return SingleLabelDataset(config, mean=self.mean, std=self.std)
        elif t == 1:
            return SequenceUnlabelDataset(config, mean=self.mean, std=self.std)
        elif t == 2:
            return DukeSeqLabelDataset(config, mean=self.mean, std=self.std)

        return None

    def try_load(self, name, config):
        dataset = self.load(config[name]) if (name in config) else None
        return dataset

    def load_dataset(self, config):
        train = self.try_load("train", config)
        unlabel = self.try_load("unlabel", config)
        val = self.try_load("val", config)
        test = self.try_load("test", config)

        if test is not None:
            return test

        if unlabel is not None:
            return (train, unlabel, val)

        return (train, val)