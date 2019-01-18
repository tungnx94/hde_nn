import os
import numpy as np
import pandas as pd 

from sequenceData import SequenceLabelDataset


class DukeSeqLabelDataset(SequenceLabelDataset):

    def __init__(self, name, data_file=None,
                 img_size=192, data_aug=False, mean=[0, 0, 0], std=[1, 1, 1], seq_length=24, saved_file=None):
        self.data_file = data_file

        SequenceLabelDataset.__init__(
            self, name, img_size, data_aug, 0, mean, std, seq_length, saved_file)

    def init_data(self):
        frame_iter = 6
        base_folder = os.path.dirname(self.data_file)
        data = pd.read_csv(self.data_file).to_dict(orient='records')

        last_idx = -1
        last_cam = -1
        seq = []  # current image sequence

        for point in data:
            img_name = os.path.basename(point['path']).strip()
            img_path = os.path.join(base_folder, point['path'])
            label = np.array(
                [point['sin'], point['cos']], dtype=np.float32)

            # extract frame id
            frame_id = int(img_name.split('_')[1][5:])
            cam_num = img_name.split('_')[0]  # camera number

            if (seq == []) or (frame_id == last_idx + frame_iter) and (cam_num == last_cam):
                seq.append((img_path, label))
                last_idx = frame_id
                last_cam = cam_num
            else:  # the index is not continuous -> save current seq
                self.save_sequence(seq)
                seq = []
                last_idx = -1
                last_cam = -1

        self.save_sequence(seq)


if __name__ == '__main__':  # test
    import sys
    sys.path.insert(0, "..")

    from generalData import DataLoader
    from utils import get_path, seq_show

    unlabelset = DukeSeqLabelDataset(
        "duke-test", data_file=get_path('DukeMTMC/val/person.csv'))
    dataloader = DataLoader(unlabelset)

    for count in range(5):
        sample = dataloader.next_sample()
        imgseq = sample[0].squeeze()
        labelseq = sample[1].squeeze()

        seq_show(imgseq.numpy(), dir_seq=labelseq)
