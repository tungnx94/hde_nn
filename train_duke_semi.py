import sysT
import torch
import random

import torch.nn as nn
import torch.optim as optim
import numpy as np
import config as cnf

from math import pi
from os.path import join
from torch.utils.data import DataLoader

from workflow import WorkFlow
from train_wf import TrainWF 
from test_wf import TestLabelSeqWF, Test

from MobileReg import MobileReg
from utils import loadPretrain, seq_show, unlabel_loss, angle_metric

from labelData import LabelDataset
from unlabelData import UnlabelDataset
from folderLabelData import FolderLabelDataset
from folderUnlabelData import FolderUnlabelDataset
from dukeSeqLabelData import DukeSeqLabelDataset

import sys
sys.path.append('../WorkFlow')

# hardcode in labelData, used where ?
train_label_file = '/datadrive/person/DukeMTMC/trainval_duke.txt'
test_label_file = '/datadrive/person/DukeMTMC/test_heading_gt.txt'
unlabel_file = 'duke_unlabeldata.pkl'
saveModelName = 'facing'

test_label_img_folder = '/home/wenshan/headingdata/val_drone'
test_unlabel_img_folder = '/datadrive/exp_bags/20180811_gascola'

pre_mobile_model = 'pretrained_models/mobilenet_v1_0.50_224.pth'

pre_model = 'models/1_2_facing_20000.pkl'

TestType = 2  # 0: none, 1: labeled sequence, 2: labeled folder, 3: unlabeled sequence

def select_WF():
    """ choose WF from test type """
    wf = GeneralWF("./", prefix=exp_prefix).

    return wf

def main():
    """ Train and validate new model """
    try:
        # Instantiate workflow.
        if wf = select_WF(TestType)
        

        wf.initialize()
        wf.run()
        wf.finalize()

    except WorkFlow.SigIntException as e:
        wf.finalize()
    except WorkFlow.WFException as e:
        print(e.describe())

if __name__ == "__main__":
    main()
