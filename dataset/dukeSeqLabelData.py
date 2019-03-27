import os
import numpy as np
import pandas as pd 

from sequenceData import SequenceLabelDataset


class DukeSeqLabelDataset(SequenceLabelDataset):

    def init_data(self):
        data_file = self.path

        frame_iter = 6
        base_folder = os.path.dirname(data_file)
        data = pd.read_csv(data_file).to_dict(orient='records')

        last_idx = -1
        last_cam = -1
        seq = []  # current image sequence

        for point in data:
            img_name = os.path.basename(point['path']).strip()
            img_path = os.path.join(base_folder, point['path'])

            angle = point['angle']
            label = np.array(
                [np.sin(angle), np.cos(angle)], dtype=np.float32)
            group = point['direction']

            # extract frame id
            frame_id = int(img_name.split('_')[1][5:])
            cam_num = img_name.split('_')[0]  # camera number

            if not ((seq == []) or ((frame_id == last_idx + frame_iter) and (cam_num == last_cam))): # split here
                # print frame_id, cam_num, len(seq)
                self.save_sequence(seq)
                seq = []

            last_idx = frame_id
            last_cam = cam_num
            seq.append((img_path, label, group))

        self.save_sequence(seq)


if __name__ == '__main__':  # test
    import sys
    sys.path.insert(0, "..")

    from generalData import DataLoader
    from utils import get_path, seq_show

    
    unlabelset = DukeSeqLabelDataset(
        "duke-test", path=get_path('DukeMTMC/test.csv'))
    dataloader = DataLoader(unlabelset)

    for count in range(5):
        sample = dataloader.next_sample()
        imgseq = sample[0].squeeze()
        labelseq = sample[1].squeeze()

        seq_show(imgseq.numpy(), dir_seq=labelseq)
