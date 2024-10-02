from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import datetime

now = datetime.datetime.now()
imgnameprefix = now.strftime("%m-%d-%Y-%H-%M-%S")
imgname = imgnameprefix + "_stitched.png"

imgpath = "../Images/53framesstaffside/"
# imgpath = "./cameraframes/"
outpath = "../Results/" + imgname

# # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--images",
    type=str,
    required=True,
    help="path to input directory of images to stitch",
)
ap.add_argument(
    "-o", "--output", type=str, required=True, help="path to the output image"
)
ap.add_argument(
    "-c",
    "--crop",
    type=int,
    default=0,
    help="whether to crop out largest rectangular region",
)
args = vars(ap.parse_args(["--images", imgpath, "--output", outpath]))

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

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

# loop over the image paths, load each one, and add them to our
# images to stich list
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    
    processedimage = imagePreProcess(image)
    images.append(processedimage)
    print(imagePath)

if __name__ == "__main__":
    # stitching
    print("[INFO] stitching images...")

    stitcher = ( 
        cv2.createStitcher(cv2.Stitcher_SCANS)
        if imutils.is_cv3()
        else cv2.Stitcher_create(cv2.Stitcher_SCANS)
    )
    (status, stitched) = stitcher.stitch(images)

    # stitching
    if status == 0:
        # write the output stitched image to disk
        print("[INFO] save stitching...")
        cv2.imwrite(args["output"], stitched)

        # display the output stitched image to our screen
        
        print("[INFO] displayed stitching...")
        cv2.namedWindow("Stitched", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Stitched", 1280, 720)
        cv2.imshow("Stitched", stitched)
        cv2.waitKey(0)

    else:
        print("[INFO] image stitching failed ({})".format(status))




















































# import cv2
# import datetime
# import glob
# import time
# from imutils import paths

# def wb_opencv(img):
#     wb = cv2.xphoto.createSimpleWB()
#     wb_img = wb.balanceWhite(img)
#     return wb_img

# def chanelClahe(channel):
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
#     channel_clahe = clahe.apply(channel)
#     return channel_clahe

# def outputImage(final_image, imgnum):
#     now = datetime.datetime.now()
#     imgdate = now.strftime("%m-%d-%Y")
    
#     imgname = imgdate + "_" + str(imgnum) + "_WB_CLAHE.png"
#     outpath = "../WB_CLAHE_IMG/" + imgname
    
#     cv2.imwrite(outpath, final_image)

# image_path = glob.glob('../Images/seaTrial30pics/*')

# images = []
# for imageName in image_path:
#     # print(image_path)
#     img = cv2.imread(imageName)
#     img = cv2.resize(img, (854 , 480))
#     images.append(img)
#     print(imageName)

# imgnameprefix = 0

# for image in images:
#     print("s")
#     white_balanced_img = wb_opencv(image)

#     lab_image = cv2.cvtColor(white_balanced_img, cv2.COLOR_BGR2LAB)
#     l_channel, a_channel, b_channel = cv2.split(lab_image)

#     l_channel_clahe = chanelClahe(l_channel)
#     merged_lab = cv2.merge((l_channel_clahe, a_channel, b_channel))

#     final_img_lab = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
#     imgnameprefix += 1
#     outputImage(final_img_lab, imgnameprefix)
    
    
# stitcher = cv2.createStitcher(cv2.Stitcher_PANORAMA)
# (status, stitched) = stitcher.stitch(images)







