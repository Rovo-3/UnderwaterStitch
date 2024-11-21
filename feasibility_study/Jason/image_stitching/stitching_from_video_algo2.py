from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
# import time
from datetime import datetime
from stitching import Stitcher

def dynamic_white_balance(image):
    wb = cv2.xphoto.createSimpleWB()
    corrected_image = wb.balanceWhite(image)
    return corrected_image

# Function for CLAHE contrast enhancement
def apply_CLAHE(image, clip_limit=2.0, grid_size=(4, 4)):
    # convert to lab colorspace and spliting it
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # increase the contrast of image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)
    # merge it
    lab_clahe = cv2.merge((l_clahe, a, b))
    output = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR) 
    return output

def stitching_image(fileName, currentImage):
    image = cv2.imread(fileName)
    try:
    # if first timer, create the image using the current image
        if image is None:
            cv2.imwrite(fileName, currentImage)
        else:
            stitcher = Stitcher(detector="akaze", confidence_threshold=0.2, crop=False)
            # stitcher = cv2.Stitcher_create(cv2.STITCHER_SCANS)
            # try:

            stitched = stitcher.stitch([image, currentImage])
            # if status == 0:
            print("Done Stitching")
            cv2.imwrite(fileName, stitched)
            cv2.imshow("stitched", stitched)
            
            # pass
            # print("Failed Stitching")
            # global frame_count 
            # frame_count=-1
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

cam =cv2.VideoCapture("./videos/VID1.mp4")
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the resolution to 1280x720
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cam.set(cv2.CAP_PROP_FPS, 10)  # Set the frame rate to 30 fps
# cam.set(cv2.CAP_PROP_EXPOSURE, -4)  # Because who needs sunlight, right?
# cam.set(cv2.CAP_PROP_GAIN, 0)       # Zero gain, zero pain!

frame_count = 0
frame_skip = 10

now = datetime.now()
date = now.strftime('%y%m%d_%H%M%S')
filename="./output/webcam/stitchedimage"+str(date)+".jpg"

while True:
    ret,cap = cam.read()
    cv2.imshow("Video",cap)
    # print(cap.shape)
    image = get_center_roi(cap,1,1)
    image = dynamic_white_balance(image)
    image=apply_CLAHE(image)
    
    if frame_count % frame_skip == 0:
        print("stitching on progress")
        stitching_image(fileName=filename, currentImage=image)
        print("stitching finished")

    frame_count+=1
    # cv2.imshow("Video",cap)
    if cv2.waitKey(1) == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
        break



# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", type=str,
# 	help="path to input directory of images to stitch")
# ap.add_argument("-o", "--output", type=str,
# 	help="path to the output image")
# ap.add_argument("-c", "--crop", type=int, default=1,
# 	help="whether to crop out largest rectangular region")
# args = vars(ap.parse_args())

# # grab the paths to the input images and initialize our images list
# print("[INFO] loading images...")
# imagePaths = sorted(list(paths.list_images("./images/set13")))
# print(imagePaths)
# images = []

# # loop over the image paths, load each one, and add them to our
# # images to stich list
# for imagePath in imagePaths:
#     image=cv2.imread(imagePath)
#     # image=dynamic_white_balance(image)
#     # image=apply_CLAHE(image)
#     image=cv2.resize(image,(0,0),fx=1, fy=1)
#     # image = cv2.undistort (image, camera_matrix, dist_coeffs)
#     images.append(image)

# print("[INFO] stitching images...")
# # ORB
# stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
# (status, stitched) = stitcher.stitch(images)
# print("done stitching")

# # if the status is '0', then OpenCV successfully performed image
# if status == 0:
#     # check to see if we supposed to crop out the largest rectangular
#     # region from the stitched image
#     date=time.time()
#     filename="./output/stitchedimage"+str(date)+".jpg"
#     print(filename)

#     print("status:", status)
#     cv2.imwrite(filename, stitched)
#     # display the output stitched image to our screen
#     cv2.imshow("Stitched", stitched)
#     cv2.waitKey(0)
# # otherwise the stitching failed, likely due to not enough keypoints)
# # being detected
# else:
#     print("[INFO] image stitching failed ({})".format(status))