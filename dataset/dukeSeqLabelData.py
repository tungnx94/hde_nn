# for dukeMCMT dataset
# return a sequence of data with label
import sys
sys.path.insert(0, "..")

import os
import cv2
import numpy as np

from utils.data import unlabel_loss, label_from_angle
from generalData import SequenceDataset


class DukeSeqLabelDataset(SequenceDataset):

    def __init__(self, label_file,
                 img_size=192, data_aug=False, mean=[0, 0, 0], std=[1, 1, 1], seq_length=32):

        self.label_file = label_file

        super(DukeSeqLabelDataset, self).__init__(img_size, data_aug, 0, mean, std, seq_length)
        self.read_debug()

    def load_image_sequences(self):
        frame_iter = 6
        # images
        img_dir = os.path.split(self.label_file)[0]  # correct?
        with open(self.label_file, 'r') as f:
            lines = f.readlines()

        last_idx = -1
        last_cam = -1
        sequence = []  # image sequence
        for line in lines:
            [img_name, angle] = line.strip().split(' ')

            # extract frame id
            frame_id = img_name.strip().split('/')[-1].split('_')[1][5:]
            try:
                frame_id = int(frame_id)
            except:
                print 'filename parse error:', img_name, frame_id
                continue

            file_path = os.path.join(img_dir, img_name)
            cam_num = img_name.strip().split('_')[0]  # what is this ?

            # import ipdb; ipdb.set_trace()
            if (last_idx < 0 or frame_id == last_idx + frame_iter) and (cam_num == last_cam or last_cam == -1):
                sequence.append((file_path, angle))
                last_idx = frame_id
                last_cam = cam_num
            else:  # the index is not continuous
                sequence = self.save_sequence(sequence)
                last_idx = -1
                last_cam = -1

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


def main():
    # test
    from generalData import DataLoader
    from utils.image import seq_show
    from utils.data import get_path
    np.set_printoptions(precision=4)

    label_file = 'DukeMCMT/test_heading_gt.txt'
    unlabelset = DukeSeqLabelDataset(
        label_file=get_path(label_file), seq_length=24, data_aug=True)
    print len(unlabelset)

    dataloader = DataLoader(unlabelset)

    count = 10
    for sample in dataloader:        
        imgseq, labelseq = sample['imgseq'].squeeze().numpy(), sample[
            'labelseq'].squeeze().numpy()
        print "unlabel loss: ", unlabel_loss(labelseq, 0.005)

        fakelabel = np.random.rand(24, 2)
        print "fake loss: ", unlabel_loss(fakelabel, 0.005)

        seq_show(imgseq, dir_seq=labelseq)

        count -= 1
        if count < 0:
            break

if __name__ == '__main__':
    main()
