import os
import numpy as np
import pandas as pd 

from sequenceData import SequenceLabelDataset

class ViratSeqLabelDataset(SequenceLabelDataset):

    def __init__(self, name, data_file, img_size=192, data_aug=False,
                 mean=[0, 0, 0], std=[1, 1, 1], seq_length=32, subsample_rate=3):

        self.data_file = data_file
        self.subsample_rate = subsample_rate

        SequenceLabelDataset.__init__(
            self, name, img_size, data_aug, 0, mean, std, seq_length, saved_file)

    def init_data(self):
        # need fixing
        img_dir = os.path.split(self.data_file)[0]  # role ?
        with open(self.data_file, 'r') as f:
            lines = f.readlines()

        line_dict = {}
        for line in lines:
            imgname, angle = line.strip().split(' ')
            img_id, frame_num = imgname.strip().split('/')
            frame_num = frame_num.split('.')[0]
            if img_id not in line_dict:
                line_dict[img_id] = []
            line_dict[img_id].append([line, int(frame_num)])

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
    import sys
    sys.path.insert(0, "..")

    from generalData import DataLoader
    from utils import get_path, seq_show

    unlabelset = ViratSeqLabelDataset("virat-train",
        data_file=get_path('VIRAT/train/person.csv'), seq_length=24)

    dataloader = DataLoader(unlabelset)
    for count in range(5):
        imgseq, labelseq = dataloader.next_sample()
        imgseq = imgseq.squeeze().numpy()
        labelseq = labelseq.squeeze().numpy()

        seq_show(imgseq, dir_seq=labelseq, scale=0.8)
