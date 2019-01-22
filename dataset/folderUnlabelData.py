# wrapper for DUKE, UCF unlabel data
import os
from os.path import join

from sequenceData import SequenceDataset, SequenceUnlabelDataset


class FolderUnlabelDataset(SequenceUnlabelDataset):

    def __init__(self, name, img_dir=None,
                 img_size=192, data_aug=False, mean=[0, 0, 0], std=[1, 1, 1],
                 seq_length=24, saved_file=None, auto_shuffle=False):

        self.img_dir = img_dir
        SequenceUnlabelDataset.__init__(self, name, img_size, data_aug, 0, mean, std, seq_length, saved_file)

    def init_data(self):
        img_folders = [folder for folder in os.listdir(
            self.img_dir) if os.path.isdir(join(self.img_dir, folder))]

        for folder in img_folders:
            folder_path = join(self.img_dir, folder)

            img_list = [file for file in os.listdir(
                folder_path) if file.endswith('.jpg') or file.endswith('png')]
            seq = [join(folder_path, img) for img in sorted(img_list)]

            self.save_sequence(seq)


if __name__ == '__main__':  # test
    import sys
    sys.path.insert(0, "..")

    from generalData import DataLoader
    from utils import get_path, seq_show

    duke = FolderUnlabelDataset("duke-unlabel", img_dir=get_path("DukeMTMC/train/images_unlabel"))
    ucf = FolderUnlabelDataset("ucf-unlabel", img_dir=get_path("UCF"))
    drone = FolderUnlabelDataset("drone-unlabel", img_dir=get_path("DRONE_seq"))

    for dataset in [duke, ucf, drone]:
        print dataset
        dataloader = DataLoader(dataset, batch_size=1)

        """
        for count in range(3):
            sample = dataloader.next_sample()
            seq_show(sample.squeeze().numpy(), scale=0.8)
        """
