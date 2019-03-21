# wrapper for DUKE, UCF unlabel data
import os
import cv2 
import numpy as np 

from os.path import join
from sequenceData import SequenceDataset

class FolderUnlabelDataset(SequenceDataset):

    def init_data(self):
        img_dir = self.path 

        img_folders = [folder for folder in os.listdir(img_dir) if os.path.isdir(join(img_dir, folder))]

        for folder in img_folders:
            folder_path = join(img_dir, folder)

            img_list = [file for file in os.listdir(
                folder_path) if file.endswith('.jpg') or file.endswith('png')]
            seq = [join(folder_path, img) for img in sorted(img_list)]

            self.save_sequence(seq)

    def __getitem__(self, idx):
        flip = self.get_flipping()

        out_seq = []
        for img_path in self.items[idx]:
            img = cv2.imread(img_path)
            out_img = self.augment_image(img, flip)
            out_seq.append(out_img)

        return np.array(out_seq)


if __name__ == '__main__':  # test
    import sys
    sys.path.insert(0, '..')

    from generalData import DataLoader
    from utils import get_path, seq_show

    duke = FolderUnlabelDataset('duke-unlabel', path=get_path('DukeMTMC/train/images_unlabel'))
    ucf = FolderUnlabelDataset('ucf-unlabel', path=get_path('UCF'))
    drone = FolderUnlabelDataset('drone-unlabel', path=get_path('DRONE_seq'))

    for dataset in [duke, ucf, drone]:
        print dataset
        dataloader = DataLoader(dataset, batch_size=1)

        for count in range(3):
            sample = dataloader.next_sample()
            seq_show(sample.squeeze().numpy(), scale=0.8)
