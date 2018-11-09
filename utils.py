import cv2
import torch
import random

import numpy as np

from math import sqrt, sin, cos
import numpy as np


def loadPretrain(model, preTrainModel):
    """ load the trained parameters from a pickle file """
    model_dict = model.state_dict()

    preTrainDict = torch.load(preTrainModel)
    preTrainDict = {k: v for k, v in preTrainDict.items() if k in model_dict}

    # debug
    for item in preTrainDict:
        print '  Load pretrained layer: ', item

    model_dict.update(preTrainDict)
    model.load_state_dict(model_dict)
    return model

# important ?
def loadPretrain2(model, preTrainModel):
    """
    load the trained parameters from a pickle file
    naming bug
    """
    preTrainDict = torch.load(preTrainModel)
    model_dict = model.state_dict()

    # update the keyname according to the last two words
    loadDict = {}
    for k, v in preTrainDict.items():
        keys = k.split('.')
        for k2, v2 in model_dict.items():
            keys2 = k2.split('.')
            if keys[-1] == keys2[-1] and (keys[-2] == keys2[-2] or
                                          (keys[-2][1:] == keys2[-2][2:] and keys[-2][0] == 'd' and keys2[-2][0:2] == 'de')):  # compensate for naming bug
                loadDict[k2] = v
                print '  Load pretrained layer: ', k2
                break

    model_dict.update(loadDict)
    model.load_state_dict(model_dict)
    return model


def getColor(x, y, maxx, maxy): #how ?
    """ :return (r,g,b,a) """

    # normalize two axis
    y = y * maxx / maxy
    maxy = maxx  

    # get red 
    x1, y1, t = x, y, maxx
    r = np.clip(1 - sqrt(float(x1 * x1 + y1 * y1)) / t, 0, 1)

    # get green
    x1, y1 = maxx - x, y
    g = np.clip(1 - sqrt(float(x1 * x1 + y1 * y1)) / t, 0, 1)

    # get blue
    x1, y1 = x, maxy - y
    b = np.clip(1 - sqrt(float(x1 * x1 + y1 * y1)) / t, 0, 1)

    # x1, y1 = maxx-x, maxy-y
    # a = sqrt(float(x1*x1+y1*y1))/t
    a = 1

    return (r, g, b, a)


# resnet: mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
def img_normalize(img, mean=[0, 0, 0], std=[1, 1, 1]):
    """ normalize RGB value to range [0..1] """
    img = img[:, :, [2, 1, 0]]  # bgr to rgb
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    img = img.transpose(2, 0, 1)
    return img

def img_denormalize(img, mean=[0, 0, 0], std=[1, 1, 1]):
    """ denormalize RGB value for visualization"""

    # print img.shape
    img = img.transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    img = img.clip(0, 1)  # network can output values out of range
    img = (img * 255).astype(np.uint8)
    img = img[:, :, [2, 1, 0]]
    return img


def seq_show(img_seq, scale=0.3):
    # input a numpy array: n x 3 x h x w
    imgnum = img_seq.shape[0]
    imgshow = []
    for k in range(imgnum):
        imgshow.append(img_denormalize(img_seq[k, :, :, :]))  # n x h x w x 3
    imgshow = np.array(imgshow)
    imgshow = imgshow.transpose(1, 0, 2, 3).reshape(
        img_seq.shape[2], -1, 3)  # h x (n x w) x 3
    imgshow = cv2.resize(imgshow, (0, 0), fx=scale, fy=scale)
    cv2.imshow('img', imgshow)
    cv2.waitKey(0)


def put_arrow(img, dir, center_x=150, center_y=96):
    """ draw an arrow on image at (center_x, center_y) """

    # print type(img), img.dtype, img.shape
    img = img.copy()

    cv2.line(img, (center_y - 30, center_x),
             (center_y + 30, center_x), (0, 255, 0), 2)

    cv2.line(img, (center_y, center_x - 30),
             (center_y, center_x + 30), (0, 255, 0), 2)

    cv2.arrowedLine(img, (center_y, center_x), (int(
        center_y + 40 * dir[1]), int(center_x - 40 * dir[0])), (0, 0, 255), 4)

    return img


def seq_show_with_arrow(img_seq, dir_seq, scale=0.8, mean=[0, 0, 0], std=[1, 1, 1]):
    """ 
    display images with arrow
    :param img_seq: a numpy array: n x 3 x h x w (images)
    :param dir_seq: a numpy array: n x 2 (directions)
    """ 

    imgnum = img_seq.shape[0]
    imgshow = []

    for k in range(imgnum):
        img = img_denormalize(img_seq[k, :, :, :], mean, std)
        img = put_arrow(img, dir_seq[k, :])
        imgshow.append(img)  # n x h x w x 3

    imgshow = np.array(imgshow)
    imgshow = imgshow.transpose(1, 0, 2, 3).reshape(
        img_seq.shape[2], -1, 3)  # h x (n x w) x 3

    imgshow = cv2.resize(imgshow, (0, 0), fx=scale, fy=scale)
    cv2.imshow('img', imgshow)
    cv2.waitKey(0)


def groupPlot(data_x, data_y, group=10):
    """ plot data by group, each using mean of coordinates """
    data_x, data_y = np.array(data_x), np.array(data_y)

    # truncate length
    d_len = len(data_x)/group * group
    data_x = data_x[0: d_len]
    data_y = data_y[0: d_len]

    data_x, data_y = data_x.reshape((-1, group)), data_y.reshape((-1, group))
    data_x, data_y = data_x.mean(axis=1), data_y.mean(axis=1)
    return (data_x, data_y)

# amigo add for data augmentation before normalization
def im_hsv_augmentation(image, Hscale=10, Sscale=60, Vscale=60):
    """ get HSV-image with noise"""

    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # introduce noise
    h = random.random() * 2 - 1
    s = random.random() * 2 - 1
    v = random.random() * 2 - 1

    imageHSV[:, :, 0] = np.clip(imageHSV[:, :, 0] + Hscale * h, 0, 255)
    imageHSV[:, :, 1] = np.clip(imageHSV[:, :, 1] + Sscale * s, 0, 255)
    imageHSV[:, :, 2] = np.clip(imageHSV[:, :, 2] + Vscale * v, 0, 255)

    # convert back to RGB
    image = cv2.cvtColor(imageHSV, cv2.COLOR_HSV2BGR)
    return image


def im_crop(image, maxscale=0.2):
    """ crop an image randomly in range [0..maxscale - (1-maxscale) .. 1] """
    shape = image.shape

    start_x = int(random.random() * maxscale * shape[1])
    end_x = int(shape[1] - random.random() * maxscale * shape[1])

    start_y = int(random.random() * maxscale * shape[0])
    end_y = int(shape[0] - random.random() * maxscale * shape[0])

    return image[start_y:end_y, start_x:end_x, :]


def im_scale_norm_pad(img, outsize=192, mean=[0, 0, 0], std=[1, 1, 1], down_reso=False, down_len=30, flip=False):
    # downsample the image for data augmentation
    minlen = np.min(img.shape[0:2])
    down_len = random.randint(down_len, down_len * 5)
    if down_reso and minlen > down_len:
        resize_scale = float(down_len) / minlen
        img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)

    resize_scale = float(outsize) / np.max(img.shape)
    # if the image is too narrow, make it more square
    miniscale = 1.8
    x_scale, y_scale = resize_scale, resize_scale
    if img.shape[0] * resize_scale < outsize / miniscale:
        y_scale = outsize / miniscale / img.shape[0]
    if img.shape[1] * resize_scale < outsize / miniscale:
        x_scale = outsize / miniscale / img.shape[1]

    img = cv2.resize(img, (0, 0), fx=x_scale, fy=y_scale)

    if flip:
        img = np.fliplr(img)

    img = img_normalize(img, mean=mean, std=std)
    # print img.shape
    imgw = img.shape[2]
    imgh = img.shape[1]
    start_x = (outsize - imgw) / 2
    start_y = (outsize - imgh) / 2
    # print start_x, start_y
    outimg = np.zeros((3, outsize, outsize), dtype=np.float32)
    outimg[:, start_y:start_y + imgh, start_x:start_x + imgw] = img

    return outimg
