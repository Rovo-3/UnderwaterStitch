from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import datetime
from natsort import natsorted

now = datetime.datetime.now()
imgnameprefix = now.strftime("%m-%d-%Y-%H-%M-%S")
imgname = imgnameprefix + "_stitched.png"

imgpath = "../Images/shelflong80frames/"
# imgpath = "./cameraframes/"
outpath = "../Results/" + imgname

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

print("[INFO] loading images...")
# imagePaths = sorted(list(paths.list_images(args["images"])))
imagePaths = natsorted(list(paths.list_images(args["images"])))
images = []
imageSizeRatio = 0.5
i = 0
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    h, w, c = image.shape
    h = h*imageSizeRatio
    w = w*imageSizeRatio
    # print (image.shape)
    print("Height: ", h, "; Width: ", w)
    image = cv2.resize(image, (854 , 480))
    # print (image.shape)
    
    processedimage = imagePreProcess(image)
    images.append(processedimage)
    i += 1
    cv2.imwrite(("./dummy/"+str(i)+".png"), processedimage)
    print(imagePath)

# if __name__ == "__main__":
#     print("[INFO] stitching images...")

#     stitcher = ( 
#         cv2.createStitcher(cv2.Stitcher_SCANS)
#         if imutils.is_cv3()
#         else cv2.Stitcher_create(cv2.Stitcher_SCANS)
#     )
#     (status, stitched) = stitcher.stitch(images)

#     if status == 0:
#         print("[INFO] save stitching...")
#         cv2.imwrite(args["output"], stitched)

#         print("[INFO] displayed stitching...")
#         cv2.namedWindow("Stitched", cv2.WINDOW_NORMAL)
#         cv2.resizeWindow("Stitched", 1280, 720)
#         cv2.imshow("Stitched", stitched)
#         cv2.waitKey(0)

#     else:
#         print("[INFO] image stitching failed ({})".format(status))