import os
import torch
import random
import json
import numpy as np

from math import pi

BASE = "/home/airlab/projects/data_icra"

ACC_THRESH = pi/8

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def get_path(data, base_folder=BASE):
    return os.path.join(base_folder, data)

def one_hot(cls):
    return np.array([int(i==cls) for i in range(8)]) 

def angle_diff(outputs, labels, mean=False):
    """ compute angular difference """
    diff = outputs- labels    
    # map to [-pi, pi]
    mask = diff < -pi
    diff[mask] = diff[mask] + 2 * pi

    mask = diff > pi
    diff[mask] = diff[mask] - 2 * pi

    diff = np.abs(diff)

    #print outputs.shape
    
    if mean:
        diff = np.mean(diff)

    return diff

def angle_diff_trigo(outputs, labels, mean=False):
    # print outputs.shape, labels.shape
    output_angle = np.arctan2(outputs[:, 0], outputs[:, 1])
    label_angle = np.arctan2(labels[:, 0], labels[:, 1])

    return angle_diff(output_angle, label_angle, mean)    

def angle_accuracy(outputs, labels):
    """ 
    compute accuracy 
    :param outputs, labels: numpy array
    """
    diff = angle_diff_trigo(outputs, labels)
    corrects = diff < ACC_THRESH

    return np.mean(corrects)

def cls_accuracy(outputs, labels):
    corrects = np.argmax(outputs, dim=1) == np.argmax(outputs, dim=1)
    return np.mean(corrects)


def angle_metric(outputs, labels):
    """ return angle loss and accuracy"""
    return angle_diff_trigo(outputs, labels, mean=True), angle_accuracy(outputs, labels)

def eval(outputs, labels):
    if type(outputs) == torch.Tensor:
        outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

    s = labels.shape

    if s[1] == 2:
        return angle_diff_trigo(outputs, labels, mean=True)
    else:
        return cls_accuracy(outputs, labels, mean=True)


def groupPlot(data_x, data_y, group=10):
    def shape_data(data):
        data = np.array(data)
        data = data[:len(data) / group * group]
        data = data.reshape((-1, group)).mean(axis=1)
        return data

    return shape_data(data_x), shape_data(data_y)
