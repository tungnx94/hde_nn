# for dukeMCMT dataset
# return a sequence of data with label

# for dukeMCMT dataset
# return a sequence of data with label
import sys
sys.path.insert(0, "..")

import os
import cv2
import numpy as np

from utils.data import unlabel_loss, label_from_angle
from generalData import SequenceDataset


class ViratSeqLabelDataset(SequenceDataset):

    def __init__(self, label_file, img_size=192, data_aug=False,
                 mean=[0, 0, 0], std=[1, 1, 1], seq_length=32, subsample_rate=3):

        self.label_file = label_file
        self.subsample_rate = subsample_rate

        super(ViratSeqLabelDataset, self).__init__(
            img_size, data_aug, 0, mean, std, seq_length)
        self.read_debug()

    def load_image_sequences(self):
        img_dir = os.path.split(label_file)[0] # role ?
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

            v = sorted(v, key=lambda x: x[1])
            split_v = []
            cur_v = [v[0]]
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

    # copied from dukeSeqLabelData
    def __getitem__(self, idx):
        ep_idx, idx = self.get_indexes(idx)

        # random fliping
        flipping = self.get_flipping()

        imgseq = []
        labelseq = []
        for k in range(self.seq_length):
            img = cv2.imread(self.img_seqs[ep_idx][idx + k][0])

            angle = self.img_seqs[ep_idx][idx + k][1]
            label = label_from_angle(angle)

            out_img, label = self.get_img_and_label(img, label, flipping)

            imgseq.append(out_img)
            labelseq.append(label)

        return {'imgseq': np.array(imgseq), 'labelseq': np.array(labelseq)}


if __name__ == '__main__':
    # test
    from generalData import DataLoader
    from utils.image import seq_show
    from utils.data import get_path
    np.set_printoptions(precision=4)

    label_file = 'VIRAT/train/annotations/annotations.csv'
    unlabelset = ViratSeqLabelDataset(
        get_path(label_file), seq_length=24, data_aug=True)
    print len(unlabelset)

    import ipdb
    ipdb.set_trace()

    dataloader = DataLoader(unlabelset)
    count = 10

    for sample in dataloader:
        imgseq, labelseq = sample['imgseq'].squeeze().numpy(), sample[
            'labelseq'].squeeze().numpy()

        seq_show(imgseq, dir_seq=labelseq, scale=0.8)

        count -= 1
        if count < 0:
            break