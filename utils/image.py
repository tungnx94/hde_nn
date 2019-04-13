import cv2
import math
import random
import numpy as np

# resnet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

def img_normalize(img, mean, std):
    """ normalize RGB value """
    #img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1) # shape = (3, width, height)
    return img


def img_denormalize(img, mean, std):
    """ denormalize RGB value for visualization"""
    # img.shape = (3, width, height)
    img = img.transpose(1, 2, 0)
    img = (img * std) + mean
    img = img.clip(0, 255).astype(np.uint8)
    #img = (img*255).clip(0, 255).astype(np.uint8)
    return img


def put_arrow(img, dir, center_x=150, center_y=96):
    """ draw an arrow on image at (center_x, center_y) """
    img = img.copy()

    cv2.line(img, (center_y - 30, center_x),
             (center_y + 30, center_x), (0, 255, 0), 2)

    cv2.line(img, (center_y, center_x - 30),
             (center_y, center_x + 30), (0, 255, 0), 2)

    cv2.arrowedLine(img, (center_y, center_x), (int(
        center_y + 40 * dir[1]), int(center_x - 40 * dir[0])), (0, 0, 255), 4)

    return img

WAIT = 1000
def seq_show(img_seq, dir_seq=None, scale=0.8, mean=[0, 0, 0], std=[1, 1, 1]):
    """ 
    display images (optional with arrow)
    :param img_seq: a numpy array: n x 3 x h x w (images)
    :param dir_seq: a numpy array: n x 2 (directions)
    :param scale:
    """
    imgnum = img_seq.shape[0]
    imgshow = []

    for k in range(imgnum):
        img = img_denormalize(img_seq[k], mean, std)

        # put arrow
        if dir_seq is not None:
            img = put_arrow(img, dir_seq[k])

        imgshow.append(img)  # n x h x w x 3

    imgshow = np.array(imgshow)
    imgshow = imgshow.transpose(1, 0, 2, 3).reshape(
        img_seq.shape[2], -1, 3)  # h x (n x w) x 3

    imgshow = cv2.resize(imgshow, (0, 0), fx=scale, fy=scale)
    cv2.imshow('img', imgshow)
    cv2.waitKey(WAIT)


# amigo add for data augmentation before normalization
def im_hsv_augmentation(image, Hscale=10, Sscale=60, Vscale=60):
    ### add noise in HSV colorspace

    # convert BGR to HSV
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

    # introduce noise factor in range [-1..1]
    h = random.random() * 2 - 1
    s = random.random() * 2 - 1
    v = random.random() * 2 - 1

    # add noise while keeping range value [0..255]
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


def im_scale_norm_pad(img, mean, std, out_size=192, flip=False):
    # apply augmentation by scale, normalization and padding
    # output a numpy array (3, width, height)

    resize_scale = float(out_size) / np.max(img.shape)
    # if the image is too narrow, make it more square
    miniscale = 1.8
    x_scale, y_scale = resize_scale, resize_scale
    if img.shape[0] * resize_scale < out_size / miniscale:
        y_scale = out_size / miniscale / img.shape[0]
    if img.shape[1] * resize_scale < out_size / miniscale:
        x_scale = out_size / miniscale / img.shape[1]

    # guarantee the longer side with be 192 pixel
    img = cv2.resize(img, (0, 0), fx=x_scale, fy=y_scale)

    # flip left-right
    if flip:
        img = np.fliplr(img)

    # normalize
    img = img_normalize(img, mean=mean, std=std)
    
    ### (RandomExpand) put to 192x192 frame with padding zeros 
    imgw = img.shape[2]
    imgh = img.shape[1]
    start_x = (out_size - imgw) // 2
    start_y = (out_size - imgh) // 2

    out_img = np.zeros((3, out_size, out_size), dtype=np.float32)

    out_img[:, start_y:start_y + imgh, start_x:start_x + imgw] = img

    return out_img

def im_scale_pad(img, out_size=192):
    # output a numpy array (width, height, 3)

    resize_scale = float(out_size) / np.max(img.shape)
    # if the image is too narrow, make it more square
    miniscale = 1.8
    x_scale, y_scale = resize_scale, resize_scale
    if img.shape[0] * resize_scale < out_size / miniscale:
        y_scale = out_size / miniscale / img.shape[0]
    if img.shape[1] * resize_scale < out_size / miniscale:
        x_scale = out_size / miniscale / img.shape[1]

    # guarantee the longer side with be 192 pixel
    img = cv2.resize(img, (0, 0), fx=x_scale, fy=y_scale)

    ### Put to 192x192 frame with padding zeros 
    imgh, imgw, _ = img.shape
    start_x = (out_size - imgw) // 2
    start_y = (out_size - imgh) // 2
    
    out_img = np.zeros((out_size, out_size, 3), dtype=np.float32)
    out_img[start_y:start_y + imgh, start_x:start_x + imgw] = img

    return out_img