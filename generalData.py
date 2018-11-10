import random

from torch.utils.data import Dataset
from utils import im_scale_norm_pad, im_crop, im_hsv_augmentation

class GeneralDataset(Dataset):

    def __init__(self):
        self.balance_factors = []
       	self.datasets = []
       	self.dataset_sizes = []

    def __len__(self):
        return sum(self.dataset_sizes)

    def __getitem__(self, idx):
        for dataset, dsize in zip(self.datasets, self.dataset_sizes):
            if idx >= dsize:
                idx -= dsize
            else:  # fidx the value
                idx = idx % len(dataset)
                return dataset[idx]

        print 'Error index:', idx
        return None


class SingleDataset(Dataset):

	def __init__(self, img_size, data_aug, maxscale, mean, std):
		self.aug = data_aug
        self.mean = mean
        self.std = std
        self.maxscale = maxscale
        self.img_size = img_size

        self.N = 0

	def __len__(self):
		return self.N

	def get_flipping(self):
		""" random fliping with probability 0.5 """
        if self.aug and random.random() > 0.5:
            return True
        return False

    def get_img_and_label(self, img, label, flipping):
    	""" :return pair of image after augmentation/scale and corresponding label """

		# augment to make more data
        if self.aug:
            img = im_hsv_augmentation(img)
            img = im_crop(img, maxscale=self.maxscale)

            if flipping and label is not None:
                label[1] = -label[1]

        out_img = im_scale_norm_pad(img,
        	out_size=self.img_size, mean=self.mean, std=self.std, down_reso=True, flip=flipping)

    	return out_img, label