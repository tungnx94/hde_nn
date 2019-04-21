# wrapper for DUKE, UCF unlabel data
import os
import cv2 
import numpy as np
import pandas as pd

from os.path import join
from .sequenceData import SequenceDataset

SAFE_DISTANCE = 10.0

class SequenceUnlabelDataset(SequenceDataset):

    def init_data(self):
        img_dir = os.path.dirname(self.path)
        data = pd.read_csv(self.path).to_dict(orient='list')
        img_folders = data["folder"]

        for folder in img_folders:
            folder_path = join(img_dir, folder)
            img_list = os.listdir(folder_path)
            seq = sorted([join(folder_path, img) for img in img_list]) # absolute paths

            self.save_sequence(seq)

    def __getitem__(self, idx):
        flip = self.get_flipping()

        out_seq = []
        for img_path in self.items[idx]:
            img = cv2.imread(img_path)
            out_img = self.augment_image(img, flip)
            out_seq.append(out_img)

        return np.array(out_seq)

# Wrapper for 3DPES
class ViratUnlabelDataset(SequenceUnlabelDataset):

    def init_data(self):
        img_dir = os.path.dirname(self.path)
        data = pd.read_csv(self.path).to_dict(orient='list')
        img_folders = data["folder"]

        for folder in img_folders:
            folder_path = join(img_dir, folder)
            img_list = os.listdir(folder_path)
            img_list = sorted(img_list, key=lambda x: int(x.split("_")[0])) # sort by frame_id

            seq = []
            last_id = -1
            last_pos = np.array([0, 0, 0, 0])

            for img in img_list:
                img_name = os.path.splitext(img)[0]
                frame_id = int(img_name.split("_")[0])

                pos = img_name.split("_")[1:]
                pos = list(map(int, pos))
                pos = np.array(pos)

                if not ((seq == []) or (frame_id==last_id+1 and np.linalg.norm(pos-last_pos) <= SAFE_DISTANCE)):
                    self.save_sequence(seq)

                last_id = frame_id 
                seq.append(os.path.join(folder_path, img))

            if len(seq) > 0:
                self.save_sequence(seq)
