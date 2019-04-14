import os
import sys
import argparse

from wf import AccumulatedValue, AccumulatedValuePlotter
from utils import read_json

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", dest="model", default=None,
                    help="model folder")
parser.add_argument("-l", dest="limit", default=-1, type=int, 
                    help="iteration limit")
args = parser.parse_args()

model = args.model
if model is None:
    sys.exit()

train_folder = os.path.join("./log", model, "train")
cnf_path = os.path.join(train_folder, "config.json")
av_path = os.path.join(train_folder, "values.csv")

config = read_json(cnf_path)

AV = AccumulatedValue(config["acvs"])
AV.load_csv(av_path)

limit = args.limit
if limit > -1:
    AV.set_limit(limit)

AVP = []
for plot in config["plots"]:
    plotter = AccumulatedValuePlotter(plot["name"], AV, plot["values"])
    AVP.append(plotter)

for avp in AVP:
    avp.write_image_final(train_folder)

