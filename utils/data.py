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
    # angle from coordiate (x, y)
    output_angle = np.arctan2(outputs[:, 0], outputs[:, 1])
    label_angle = np.arctan2(labels[:, 0], labels[:, 1])

    diff_angle = output_angle - label_angle
    # map to [-pi, pi]
    mask = diff_angle < -pi
    diff_angle[mask] = diff_angle[mask] + 2 * pi

    mask = diff_angle > pi
    diff_angle[mask] = diff_angle[mask] - 2 * pi

    diff = np.abs(diff_angle)
    if mean:
        diff = np.mean(diff)
    return diff


def angle_accuracy(outputs, labels):
    """ 
    compute accuracy 
    :param outputs, labels: numpy array
    """
    diff_angle = angle_diff(outputs, labels)
    acc_angle = diff_angle < ACC_THRESH

    acc = float(np.sum(acc_angle)) / labels.shape[0]
    return acc

def cls_accuracy(outputs, labels):
    corrects = np.argmax(outputs, dim=1) == np.argmax(outputs, dim=1)
    return np.mean(corrects)


def angle_metric(outputs, labels):
    """ return angle loss and accuracy"""
    return angle_loss(outputs, labels), accuracy_cls(outputs, labels)


def groupPlot(data_x, data_y, group=10):
    def shape_data(data):
        data = np.array(data)
        data = data[:len(data) / group * group]
        data = data.reshape((-1, group)).mean(axis=1)
        return data

    return shape_data(data_x), shape_data(data_y)
