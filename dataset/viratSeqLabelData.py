# for VIRAT dataset
# return a sequence of data with label

import sys
sys.path.insert(0, "..")

import os
import numpy as np

from generalData import SingleSequenceDataset

# not yet functional 
# needs further digging
class ViratSeqLabelDataset(SingleSequenceDataset):

    def __init__(self, name, label_file, img_size=192, data_aug=False,
                 mean=[0, 0, 0], std=[1, 1, 1], seq_length=32, subsample_rate=3):

        self.label_file = label_file
        self.subsample_rate = subsample_rate #distance between 2 frame

        super(ViratSeqLabelDataset, self).__init__(name, img_size, data_aug, 0, mean, std, seq_length)
        self.read_debug()

    def load_image_sequences(self):
        # need fixing
        img_dir = os.path.split(self.label_file)[0]  # role ?
        with open(self.label_file, 'r') as f:
            lines = f.readlines()

        line_dict = {}
        for line in lines:
            imgname, angle = line.strip().split(' ')
            img_id, frame_num = imgname.strip().split('/')
            frame_num = frame_num.split('.')[0]
            if img_id not in line_dict:
                line_dict[img_id] = []
            line_dict[img_id].append([line, int(frame_num)])

        # ipdb.set_trace() # check the dict
        sequencelist = []
        for k, v in line_dict.items():
            if len(v) <= 1:
                continue

            # v = [[line, frame_num]]
            v = sorted(v, key=lambda x: x[1]) #sort by frame number
            split_v = [] # frame sequences 
            cur_v = [v[0]] # consecutive frames
            for i in range(1, len(v)):
                if v[i][1] == v[i - 1][1] + 1:
                    cur_v.append(v[i])
                else:
                    split_v.append(cur_v)
                    cur_v = [v[i]]
            if len(cur_v):
                split_v.append(cur_v)

            # subsample
            for x in split_v:
                idx = np.arange(0, len(x), subsample_rate)
                subsampled_arr = np.array(x)[idx]
                if len(subsampled_arr) >= seq_length:
                    subsampled_arr = [(join(self.img_dir, x[0].strip().split(' ')[0]), x[
                                       0].strip().split(' ')[1]) for x in subsampled_arr]
                    self.img_seqs.append(subsampled_arr)


if __name__ == '__main__':
    from generalData import DataLoader
    from utils import get_path, seq_show

    unlabelset = ViratSeqLabelDataset("virat-train",
        get_path('VIRAT/train/annotations/annotations.csv'), seq_length=24, data_aug=True)

    dataloader = DataLoader(unlabelset)


    for count in range(10):
        sample = dataloader.next_sample()

        imgseq, labelseq = sample['imgseq'].squeeze().numpy(), sample[
            'labelseq'].squeeze().numpy()

        seq_show(imgseq, dir_seq=labelseq, scale=0.8)
