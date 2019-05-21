import sys
import argparse
import imutils
import time
import cv2
import os
import numpy as np

from imutils.video import FPS
from imutils.video import FileVideoStream

import utils 
from network import ModelLoader 
import torch


### Load model
config = utils.read_json("demo.json")
loader = ModelLoader()
predictor, _ = loader.load(config["model"])
print(predictor)
predictor.eval()


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

BASE = "/home/tung/projects/hde_data"

model = BASE + "/MobileNetSSD_deploy.caffemodel"
prototxt = BASE + "/MobileNetSSD_deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

def put_arrow(img, dir, center_x=150, center_y=96):
    """ draw an arrow on image at (center_x, center_y) """
    #img = img.copy()

    print("aha")
    cv2.line(img, (center_y - 30, center_x),
             (center_y + 30, center_x), (0, 255, 0), 2)

    cv2.line(img, (center_y, center_x - 30),
             (center_y, center_x + 30), (0, 255, 0), 2)

    cv2.arrowedLine(img, (center_y, center_x), (int(
        center_y + 40 * dir[1]), int(center_x - 40 * dir[0])), (0, 0, 255), 4)

def detect(image, confidence_thres=0.8, display_label=True):
    ### detect and mark a person bb in image, return the marked image
    (h, w) = image.shape[:2]
    # print(h, w)

    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_thres:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            
            if CLASSES[idx] == "person":
                print("hit")

                mX = (endX - startX) // 10
                mY = (endY - startY) // 10
                startX = max(startX - mX, 0)
                endX = min(endX + mX, w-1)
                startY = max(startY - mY, 0)
                endY = min(endY + mY, h-1)

                # get direction
                subimg = image[startY:endY, startX:endX]
                subimg = utils.im_scale_norm_pad(subimg, mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])

                inp = torch.tensor(np.array([subimg]))
                angle = predictor.forward(inp)
                #print(angle)

                angle = angle.data[0]
                alpha = np.array(angle)
                print(alpha)
                
                #print(subimg)
                #print(subimg.shape)
                #print(angle)
                

                # display bounding box
                subimg = utils.img_denormalize(subimg, [127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
                put_arrow(subimg, alpha)
                cv2.imshow("Frame", subimg)
                key = cv2.waitKey(5)
                #image[startY:endY, startX:endX] = subimg
                #image = subimg
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
            

    return image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to video")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")

confidence=0.8

args = vars(ap.parse_args())

video_path = args["video"]

print("[INFO] starting video stream...")
fvs = FileVideoStream(video_path).start()
time.sleep(1.0)
fps = FPS().start()


# loop over the frames from the video stream
count = 0
try:
    while fvs.more():
        # grab the frame from the threaded stream (and resize)
        frame = fvs.read()
        count += 1
        if count % 5 == 0:
            # frame = imutils.resize(frame, width=400)

            frame = detect(frame, confidence)

            # show the output frame & update FPS counter

            #cv2.imshow("Frame", frame)
            #key = cv2.waitKey(5)
            time.sleep(0.01)
            fps.update()
except:
    print("finished")

fps.stop()
# cleaup
cv2.destroyAllWindows()
fvs.stop()