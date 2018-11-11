# for dukeMCMT dataset
# return a sequence of data with label

import os
import cv2
import numpy as np

from utils import unlabel_loss, label_from_angle
from generalData import SequenceDataset


class DukeSeqLabelDataset(SequenceDataset):

    def __init__(self, label_file='/datadrive/person/DukeMTMC/heading_gt.txt',
                 img_size=192, data_aug=False, mean=[0, 0, 0], std=[1, 1, 1], batch=32):

        super(DukeSeqLabelDataset, self)__init__(img_size, data_aug, 0, mean, std)
        self.label_file = label_file

        self.read_debug()

    def load_image_sequences(self):
        frame_iter = 6
        # images
        img_dir = os.path.split(self.label_file)[0]  # correct?
        with open(self.label_file, 'r') as f:
            lines = f.readlines()

        last_idx = -1
        last_cam = -1
        sequence = [] #image sequence
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
            cam_num = img_name.strip().split('_')[0] # what is this ?

            # import ipdb; ipdb.set_trace()
            if (last_idx < 0 or frame_id == last_idx + frame_iter) and (cam_num == last_cam or last_cam == -1):
                sequence.append((file_path, angle))
                last_idx = frame_id
                last_cam = cam_num
            else:  # the index is not continuous
                if len(sequence) >= batch:
                    self.img_seqs.append(sequence)
                    print '** sequence: ', len(sequence)
                    sequence = []
                else:
                    print '!sequence too short'
                last_idx = -1
                last_cam = -1

    def __getitem__(self, idx):
        ep_idx, idx = self.get_indexes(idx)

        # random fliping
        flipping = self.get_flipping()

        imgseq = []
        labelseq = []
        for k in range(self.batch):
            img = cv2.imread(self.img_seqs[ep_idx][idx + k][0])

            angle = self.img_seqs[ep_idx][idx + k][1]
            label = label_from_angle(angle)

            out_img, label = self.get_img_and_label(img, label, flipping)

            imgseq.append(out_img)
            labelseq.append(label)

        return {'imgseq': np.array(imgseq), 'labelseq': np.array(labelseq)}

def main():
    # test
    from torch.utils.data import DataLoader
    from utils import seq_show
    np.set_printoptions(precision=4)

    # unlabelset = FolderUnlabelDataset(img_dir='/datadrive/person/DukeMTMC/heading',batch = 32, data_aug=True, include_all=True,datafile='duke_unlabeldata.pkl')
    # unlabelset = FolderUnlabelDataset(img_dir='/datadrive/person/DukeMTMC/heading',batch = 24, data_aug=True, include_all=True)
    unlabelset = DukeSeqLabelDataset(
        label_file='/datadrive/person/DukeMTMC/test_heading_gt.txt', batch=24, data_aug=True)
    print len(unlabelset)

    dataloader = DataLoader(unlabelset, batch_size=1,
                            shuffle=True, num_workers=1)
    data_iter = iter(dataloader)

    while True:
        try:
            sample = data_iter.next()
        except:
            data_iter = iter(dataloader)
            sample = data_iter.next()

        imgseq, labelseq = sample['imgseq'].squeeze().numpy(), sample[
            'labelseq'].squeeze().numpy()
        print "unlabel loss: ", unlabel_loss(labelseq, 0.005)

        fakelabel = np.random.rand(24, 2)
        print "fake loss: ", unlabel_loss(fakelabel, 0.005)

        seq_show(imgseq, dir_seq=labelseq)

    # import ipdb; ipdb.set_trace()
    """
    for k in range(10):
        sample = unlabelset[k*1000]
        imgseq, labelseq = sample['imgseq'], sample['labelseq']
        print imgseq.dtype, imgseq.shape
        seq_show_with_arrow(imgseq,labelseq, scale=0.8)
    """

if __name__ == '__main__':
    main()
