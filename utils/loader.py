import sys
sys.path.insert(0, '..')

from dataset import *
from network import *
from data import get_path

Thresh = 0.005  # unlabel_loss threshold

class ModelLoader(object):

    def __init__(self):
        self.name = "Model-Loader"

    def load(self, modelType, mobileNet=None):
        # 0: Vanilla, 1: MobileRNN, 2: MobileReg, 3: MobileEncoderReg
        if modelType == 2:
            model = MobileReg(lamb=0.1, thresh=Thresh)
        elif modelType == 3:
            model = MobileEncoderReg(lamb=0.001)

        if mobileNet is not None:
            model.load_mobilenet(mobileNet)
            print "Loaded MobileNet ", mobileNet

        return model

    def load_trained(self, modelType, trained_params):
        model = self.load(modelType) 
        model.load_pretrained(trained_params)

        print "Loaded weights from ", trained_params
        return model

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
        return FolderUnlabelDataset(name, img_dir=get_path(path), data_aug=data_aug, mean=self.mean, std=self.std)

    def single_label(self, name, path, data_aug=True):
        return SingleLabelDataset(name, data_file=get_path(path), data_aug=data_aug, mean=self.mean, std=self.std)

    def duke_seq(self, name, path, seq_length, data_aug=True):
        return DukeSeqLabelDataset(name, data_file=get_path(path), seq_length=seq_length, data_aug=data_aug, mean=self.mean, std=self.std)

    def virat_seq(self, name, path, seq_length, data_aug=True):
        return ViratSeqLabelDataset(name, data_file=get_path(path), seq_length=seq_length, data_aug=data_aug, mean=self.mean, std=self.std)

    def mix(self, name, sets, factors=None):
        mixset = MixDataset(name)

        if factors == None:
            factors = [1] * len(sets)

        for dts, factor in zip(sets, factors):
            mixset.add(dts, factor) 

        return mixset
