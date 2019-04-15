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

# ModelType 0: Vanilla, 1: MobileRNN, 2: MobileReg
config = read_json(args.cnf)

def select_WF():
    # avoid multiple instance of logger in WorkFlow
    WFType = config["type"]

    if WFType == "train":
        net_type = config["model"]["type"]

        if net_type in [0, 1, 3, 4]:
            return TrainSLWF(config)
        elif net_type == 2:
            return TrainSSWF(config)
    elif WFType == "test":
        testType = config["test"]
        if testType == "supervised":
            return TestLabelWF(config)
        elif testType == "semi":
            return TestLabelSeqWF(config)

    return None
    
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
