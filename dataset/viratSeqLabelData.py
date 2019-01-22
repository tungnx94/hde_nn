import sys
sys.path.insert(0, "..")

import os
import numpy as np
import pandas as pd

from sequenceData import SequenceLabelDataset
from utils import one_hot

SAFE_DISTANCE = 40.0

class ViratSeqLabelDataset(SequenceLabelDataset):

    def __init__(self, name, data_file=None, 
                img_size=192, data_aug=False, mean=[0, 0, 0], std=[1, 1, 1], seq_length=24, saved_file=None):

        self.data_file = data_file

        SequenceLabelDataset.__init__(
            self, name, img_size, data_aug, 0, mean, std, seq_length, saved_file)

    def init_data(self):

        data = pd.read_csv(self.data_file).to_dict(orient='records')
        base_folder = os.path.dirname(self.data_file)
        # each element is (image, label, direction)

        line_dict = {}

        ind = 0
        for point in data:
            img_path = os.path.join(base_folder, point['path'])
            label = np.array(
                [point['sin'], point['cos']], dtype=np.float32)
            d = one_hot(point['direction'])
            entry = (img_path, label, d)

            parts = os.path.basename(img_path).split('.')[0].split('_')
            f_index = parts.index('person') - 3

            key = parts[:f_index]
            key.append(parts[f_index+1])
            key = ''.join(key) 
            # key = ''.join(parts[:self.f_index].append(parts[self.f_index+1]))
            frame_id = int(parts[f_index])
            position = np.array([int(t) for t in parts[-4:]])

            if len(position) != 4:
                print 'error', ind, position, parts
                return

            if key not in line_dict:
                line_dict[key] = []
            else:
                line_dict[key].append((entry, frame_id, position))

            ind += 1
            # print ind

        ind = 0
        for key, line in line_dict.items():
            line = sorted(line, key=lambda x: x[1]) #sort by frame id

            seq = [] 
            last_id = 0
            last_pos = np.array([0, 0, 0, 0])
            for entry, frame_id, frame_pos in line:
                if not ((seq == []) or (frame_id==last_id+1 and np.linalg.norm(frame_pos-last_pos) <= SAFE_DISTANCE)):

                    #print len(seq), len(line)
                    self.save_sequence(seq)
                    seq = []

                last_id = frame_id
                last_pos = frame_pos
                seq.append(entry)
            
            self.save_sequence(seq)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, "..")

    from generalData import DataLoader
    from utils import get_path, seq_show

    virat = ViratSeqLabelDataset("virat",
            data_file=get_path('VIRAT/person/train.csv'), seq_length=24)

    pes = ViratSeqLabelDataset("3dpes",
            data_file=get_path('3DPES/train.csv'), seq_length=24)

    for dataset in [virat, pes]:
        print dataset
        dataloader = DataLoader(dataset, batch_size=1)

        for count in range(3):
            imgseq, angleseq, _ = dataloader.next_sample()
            imgseq = imgseq.squeeze().numpy()
            angleseq = angleseq.squeeze().numpy()

            seq_show(imgseq, dir_seq=angleseq, scale=0.8)

    """
    virat = ViratSeqLabelDataset("virat-train",
            data_file=get_path('VIRAT/person/train.csv'), seq_length=24)
    virat = ViratSeqLabelDataset("virat-val",
            data_file=get_path('VIRAT/person/val.csv'), seq_length=24)
    virat = ViratSeqLabelDataset("virat-test",
            data_file=get_path('VIRAT/person/test.csv'), seq_length=24)

    pes = ViratSeqLabelDataset("3dpes-train",
            data_file=get_path('3DPES/train.csv'), seq_length=24)
    pes = ViratSeqLabelDataset("3dpes-val",
            data_file=get_path('3DPES/val.csv'), seq_length=24)
    pes = ViratSeqLabelDataset("3dpes-test",
            data_file=get_path('3DPES/test.csv'), seq_length=24)
    """
