import os
import torch
import cv2
import random
import json
import numpy as np

from math import pi
from .image import im_scale_pad

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
        json.dump(data, f, indent=4)

def get_path(data, base_folder=BASE):
    return os.path.join(base_folder, data)

def split_half(L):
    """ split a list L in 2 halves """
    first  = L[: len(L)//2]
    second = L[len(L)//2 :]
    return first, second

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

def calculate_mean_std(img_paths, scale=True):
    count = len(img_paths) * 192 * 192

    # calculate mean
    mean = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if scale:
            img = im_scale_pad(img)

        im_mean = np.mean(img, axis=(0, 1))
        mean.append(im_mean)

    mean = np.mean(np.array(mean), axis=0)

    # calculate std
    std = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if scale:
            img = im_scale_pad(img)

        sqr_diff = (img - mean) ** 2
        std.append(np.sum(sqr_diff, axis=(0, 1)))

    std = np.sqrt(np.sum(std, axis=0) / (count-1)) 

    return (mean, std)