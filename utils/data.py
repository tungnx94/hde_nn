import os
import torch
import random
import json
import numpy as np

from math import pi

#BASE = "/home/tungnguyen/projects/data_icra"
BASE = "/home/tung/projects/data_icra"

ACC_THRESH = pi/8

def create_folder(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

def read_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data

def write_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f)

def get_path(data, base_folder=BASE):
    return os.path.join(base_folder, data)

def detach_to_numpy(data):
    if type(data) == torch.Tensor:
        data = data.cpu().detach().numpy()
    return data

def angle_diff(outputs, labels, mean=False):
    """ compute angular difference in radiance"""
    diff = outputs- labels    
    # map to [-pi, pi]
    mask = diff < -pi
    diff[mask] = diff[mask] + 2 * pi

    mask = diff > pi
    diff[mask] = diff[mask] - 2 * pi

    diff = np.abs(diff)
    if mean:
        diff = np.mean(diff)

    return diff

def angle_diff_rad(outputs, labels, mean=False):
    output_angle = np.arctan2(outputs[..., 0], outputs[..., 1])
    label_angle = np.arctan2(labels[..., 0], labels[..., 1])

    return angle_diff(output_angle, label_angle, mean)    

def angle_err(outputs, labels):
    outputs = detach_to_numpy(outputs)
    labels = detach_to_numpy(labels)

    return angle_diff_rad(outputs, labels, mean=True)

# error and accuracy
def angle_metric(outputs, labels):
    outputs = detach_to_numpy(outputs)
    labels = detach_to_numpy(labels)

    diff = angle_diff_rad(outputs, labels)
    corrects = diff < ACC_THRESH

    return np.mean(diff), np.mean(corrects)
