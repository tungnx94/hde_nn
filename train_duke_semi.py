import os 
import argparse

from wf import *
from dataset import DatasetLoader
from utils import read_json

d_loader = DatasetLoader()

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", dest="cnf", default="./config/train0a.json",
                    help="train/test config")
args = parser.parse_args()

# ModelType 0: Vanilla, 1: MobileRNN, 2: MobileReg, 3: MobileEncoderReg
# WFType 0: train, 1: labeled image, 2: unlabeled sequence, 3: labeled seq
config = read_json(args.cnf)

def select_WF():
    # avoid multiple instance of logger in WorkFlow
    WFType = config["type"]

    if WFType == "train":
        net_type = config["model"]["type"]

        if net_type in [0, 1, 3]:
            return TrainSLWF(config)
        elif net_type == 2:
            return TrainSSWF(config)
    elif WFType == "test":
        dts_type = config["dataset"]["test"]["type"]

        if dts_type == 0:
            return TestLabelWF(config)
        elif dts_type == 1:
            return TestUnlabelWF(config)
        elif dts_type == 2:
            return TestLabelSeqWF(config)

def main():
    """ Train and validate new model """
    try:
        wf = select_WF()
        wf.prepare_dataset(d_loader)
        wf.proceed()

    except WFException as e:
        print("exception", e)
        wf.finalize()

if __name__ == "__main__":
    main()
