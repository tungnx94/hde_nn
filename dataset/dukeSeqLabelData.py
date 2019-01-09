# for dukeMCMT dataset
# return a sequence of data with label
import sys
sys.path.insert(0, "..")

import os
import numpy as np

from generalData import SingleSequenceDataset

# segmentation not quite clear 
# needs further digging
class DukeSeqLabelDataset(SingleSequenceDataset):

    def __init__(self, name, label_file,
                 img_size=192, data_aug=False, mean=[0, 0, 0], std=[1, 1, 1], seq_length=32):

        self.label_file = label_file

        super(DukeSeqLabelDataset, self).__init__(
            name, img_size, data_aug, 0, mean, std, seq_length)

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


if __name__ == '__main__': # test
    from network import MobileReg
    from generalData import DataLoader
    from utils import get_path, seq_show

    unlabelset = DukeSeqLabelDataset("duke-test",
        label_file=get_path('DukeMCMT/val/val.txt'), seq_length=24, data_aug=True)

    dataloader = DataLoader(unlabelset)
    model = MobileReg()

    count = 10
    for count in range(10:)
        sample = dataloader.next_sample()
        imgseq = sample['imgseq'].squeeze()
        labelseq = sample['labelseq'].squeeze().numpy()

        loss = model.unlabel_loss(imgseq, 0.005).to("cpu").numpy()
        print "unlabel loss: ", loss

        seq_show(imgseq.numpy(), dir_seq=labelseq)
