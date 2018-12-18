import os
import torch
import random
import numpy as np

from math import pi

BASE = "/media/mohammad/DATA/icra2019"


def get_path(data, base_folder=BASE):
    return os.path.join(base_folder, data)


def new_variable(tensor, **kwargs):
    var = torch.autograd.Variable(tensor, **kwargs)
    if torch.cuda.is_available():
        var = var.cuda()
    return var
    

def unlabel_loss_np(output, threshold):
    """
    :param output: network unlabel output (converted to numpy)
    :return: unlabel loss
    """
    unlabel_batch = output.shape[0]
    loss_unlabel = 0

    for ind1 in range(unlabel_batch - 5):  # try to make every sample contribute
        # randomly pick two other samples
        ind2 = random.randint(ind1 + 2, unlabel_batch - 1)  # big distance
        ind3 = random.randint(ind1 + 1, ind2 - 1)  # small distance

        diff_big = np.sum((output[ind1] - output[ind2]) ** 2) / 2.0
        diff_small = np.sum((output[ind1] - output[ind3]) ** 2) / 2.0

        cost = max(diff_small - diff_big - threshold, 0)
        loss_unlabel += cost

    return loss_unlabel


def label_from_angle(angle):
    angle_cos = np.cos(float(angle))
    angle_sin = np.sin(float(angle))

    return np.array([angle_sin, angle_cos], dtype=np.float32)


def angle_diff(outputs, labels):
    """ compute angular difference """

    # calculate angle from coordiate (x, y)
    output_angle = np.arctan2(outputs[:, 0], outputs[:, 1])
    label_angle = np.arctan2(labels[:, 0], labels[:, 1])

    diff_angle = output_angle - label_angle

    # map to (-pi, pi)
    mask = diff_angle < -pi
    diff_angle[mask] = diff_angle[mask] + 2 * pi

    mask = diff_angle > pi
    diff_angle[mask] = diff_angle[mask] - 2 * pi

    return diff_angle


def angle_loss(outputs, labels):
    """ compute mean angular difference between outputs & labels"""
    diff_angle = angle_diff(outputs, labels)
    return np.mean(np.abs(diff_angle))


def accuracy_cls(outputs, labels):
    """ 
    compute accuracy 
    :param outputs, labels: numpy array
    """
    diff_angle = angle_diff(outputs, labels)
    acc_angle = diff_angle < 0.3927  # 22.5 * pi / 180 = pi/8

    acc = float(np.sum(acc_angle)) / labels.shape[0]
    return acc


def angle_metric(outputs, labels):
    """ return angle loss and accuracy"""
    return angle_loss(outputs, labels), angle_cls(outputs, labels)


def groupPlot(data_x, data_y, group=10):
    def shape_data(data):
        data = np.array(data)
        data = data[:len(data) / group * group]
        data = data.reshape((-1, group)).mean(axis=1)
        return data

    return shape_data(data_x), shape_data(data_y)
