from stitching import Stitcher
import numpy as np
import imutils
import cv2
import datetime
from natsort import natsorted
import glob

outpath = "../Results/" + datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + "_stitched.png"

def wb_opencv(img):
    wb = cv2.xphoto.createSimpleWB()
    wb_img = wb.balanceWhite(img)
    return wb_img

def chanelClahe(channel):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    channel_clahe = clahe.apply(channel)
    return channel_clahe

def imagePreProcess(imagetobeprocessed):
    white_balanced_img = wb_opencv(imagetobeprocessed)

    lab_image = cv2.cvtColor(white_balanced_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    l_channel_clahe = chanelClahe(l_channel)
    merged_lab = cv2.merge((l_channel_clahe, a_channel, b_channel))

    final_img_lab = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return final_img_lab

def resizeImage(img, scale):
    h, w, _ = img.shape
    h = h*scale
    w = w*scale
    # print (image.shape)
    # print("Height: ", h, "; Width: ", w)
    img = cv2.resize(img, (int(w) , int(h)))
    return img

print("[INFO] loading images...")
imagePaths = natsorted(list(glob.glob("../Images/27framesstaffside/*")))

if __name__ == "__main__":
    images = []
    
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = resizeImage(image, 1)
        
        processedimage = imagePreProcess(image)
        images.append(processedimage)
        print(imagePath)
    
    print("[INFO] stitching images...")

    (status, stitched) = cv2.Stitcher_create(cv2.Stitcher_SCANS).stitch(images)
    
    # settings = {"detector": "orb", 
    #         "confidence_threshold": 0.5, 
    #         # "blender_type": "feather",
    #         "crop": False,
    #         # "no-crop": "True",
    #         # "wave_correct_kind": "auto"
    #         "try_use_gpu": True
    #         }
    # stitcher = Stitcher(**settings)
    # stitched = stitcher.stitch(images)
    
    if status == 0:
        print("[INFO] save stitching...")
        cv2.imwrite(outpath, stitched)

        print("[INFO] displayed stitching...")
        cv2.namedWindow("Stitched", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Stitched", 1280, 720)
        cv2.imshow("Stitched", stitched)
        cv2.waitKey(0)

    else:
        print("[INFO] image stitching failed ({})".format(status))