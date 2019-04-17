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

# ModelType 0: Vanilla, 1: MobileRNN, 2: MobileReg, 3: HDE_RNN , 4: F_RNN, 5: FG_RNN, 6: MobileReg2
config = read_json(args.cnf)

def select_WF():
    # avoid multiple instance of logger in WorkFlow
    WFType = config["type"]

    if WFType == "train":
        net_type = config["model"]["type"]
        if net_type == 2:
            return TrainSSWF(config)
        elif net_type == 6:
            return TrainSSWF2(config)
        else:
            return TrainSLWF(config)
            
    elif "test" in WFType:
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
