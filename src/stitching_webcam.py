from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

import cv2.cuda as cuda
# import time
from datetime import datetime

def check_blur(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def dynamic_white_balance(image):
    wb = cv2.xphoto.createSimpleWB()
    corrected_image = wb.balanceWhite(image)
    # cv2.imwrite("ImprovedSIFT_correctedWB.jpg", corrected_image)
    # showimg("correctedWB", corrected_image)
    return corrected_image

# Function for CLAHE contrast enhancement
def apply_CLAHE(image, clip_limit=2.0, grid_size=(8, 8)):
    # convert to lab colorspace and spliting it
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # increase the contrast of image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)
    # merge it
    lab_clahe = cv2.merge((l_clahe, a, b))
    output = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR) 
    # clahe_img = cv2.imwrite("ImprovedSIFT_clahe.jpg",output)
    # showimg("output", output)
    return output

def stitching_image(fileName, currentImage):
    image = cv2.imread(fileName)
    try:
    # if first timer, create the image using the current image
        if image is None:
            cv2.imwrite(fileName, currentImage)
        else:
            stitcher = cv2.Stitcher.create(cv2.STITCHER_SCANS)
            (status, stitched) = stitcher.stitch([currentImage, image])
            print("done stitching")
            if status == 0:
                cv2.imwrite(fileName, stitched)
                stitched = cv2.resize(stitched, (0,0), fx=0.5,fy=0.5)
                cv2.imshow("stitched:", stitched)
    except Exception as e:
        print(f"Stitching failed: {e}")
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

cam =cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the resolution to 1280x720
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 10)  # Set the frame rate to 30 fps
cam.set(cv2.CAP_PROP_EXPOSURE, -4)  # Because who needs sunlight, right?
cam.set(cv2.CAP_PROP_GAIN, 0)       # Zero gain, zero pain!

frame_count = 0
frame_skip = 10

now = datetime.now()
date = now.strftime('%y%m%d_%H%M%S')
filename="./output/webcam/stitchedimage"+str(date)+".jpg"

while True:
    ret,cap = cam.read()
    # print(cap.shape)
    image = get_center_roi(cap,1,1)
    # image=dynamic_white_balance(image)
    # image=apply_CLAHE(image)
    blur_level = check_blur(image)

    if blur_level<180:
        frame_count=0
        continue
    
    if frame_count % frame_skip == 0:
        stitching_image(filename, image)

    frame_count+=1
    cap=cv2.resize(cap, (0,0),fx=0.5,fy=0.5)
    cv2.imshow("Camera",cap)
    if cv2.waitKey(1) == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
        break