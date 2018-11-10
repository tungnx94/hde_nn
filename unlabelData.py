# Combine two labeled dataset together
from generalData import GeneralDataset
from folderUnlabelData import FolderUnlabelDataset


class UnlabelDataset(Generalataset):

    def __init__(self, batch, balance=False, mean=[0, 0, 0], std=[1, 1, 1]):
        super.(LabelDataset, self)__init__()

        if balance:
            self.balance_factors = [4, 1]
        else:
            self.balance_factors = [1, 1]

        # datasets
        ucf = FolderUnlabelDataset(
            batch=batch, data_aug=True, data_file='ucf_unlabeldata.pkl', mean=mean, std=std)  # 940
        duke = FolderUnlabelDataset(
            batch=batch, data_aug=True, data_file='duke_unlabeldata.pkl', mean=mean, std=std)  # 3997

        self.datasets = [ucf, duke]

        self.dataset_sizes = [
            len(dataset) * factor for dataset, factor in zip(self.datasets, self.balance_factors)]


def main():
    # test
    import cv2
    import numpy as np
    from utils import seq_show, put_arrow
    from torch.utils.data import DataLoader

    np.set_printoptions(precision=4)

    unlabeldataset = UnlabelDataset(batch=24, balance=True)
    dataloader = DataLoader(unlabeldataset, batch_size=1,
                            shuffle=True, num_workers=1)

    # import ipdb;ipdb.set_trace()
    print len(unlabeldataset)
    for sample in dataloader:
        seq_show(sample.squeeze().numpy())

    """
    # datalist=[0,69679,69680,69680*2-1,69680*2,364785,364786]
    for k in dataloader:
        sample = labeldataset[k]
        img = sample['img']
        label = sample['label']
        print img.dtype, label
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        img = img_denormalize(img)
        img = put_arrow(img, label)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    """

if __name__ == '__main__':
    main()
