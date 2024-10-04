from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
# import time
from datetime import datetime

def get_center_roi(image, roi_width_ratio=0.8, roi_height_ratio=0.8):
    # Get the dimensions of the image
    (h, w) = image.shape[:2]

    # Calculate the width and height of the ROI
    roi_w = int(w * roi_width_ratio)
    roi_h = int(h * roi_height_ratio)

    # Calculate the top-left corner of the ROI (center it)
    x_start = (w - roi_w) // 2
    y_start = (h - roi_h) // 2

    # Extract the ROI (Region of Interest)
    roi = image[y_start:y_start + roi_h, x_start:x_start + roi_w]
    
    return roi

vid_name = "VID2"
saved_path = "./images/generated_frames/" + vid_name
vid_path = "./videos/" + vid_name + ".mp4"
cam =cv2.VideoCapture(vid_path)
frame_count = 0
frame_skip = 1

try:
    os.makedirs(saved_path)
    print(f"Directory '{saved_path}' created")
except FileExistsError:
    print(f"Directory '{saved_path}' already exists")
# print("Doing next!")
i = 1

while True:
    # if i<10:
    #     number = "00"+str(i)
    # elif i/10 < 10:
    #     number = "0"+str(i)
    # elif i/10 < 10:
    #     number = +str(i)
    try:
        ret,cap = cam.read()
        image = cap
        now = datetime.now()
        date = now.strftime('%y%m%d_%H%M%S')
        filename=saved_path+"/"+str(i)+".png"

        if frame_count % frame_skip == 0:
            cv2.imwrite(filename, image)
            print(filename)
            i+=1
        frame_count+=1
    except:
        print("fail!")
        break