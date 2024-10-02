# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import datetime


now = datetime.datetime.now()
imgnameprefix = now.strftime("%m-%d-%Y-%H-%M-%S")
imgname = imgnameprefix + "_stitched.png"

imgpath = "../WB_CLAHE_IMG/"
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

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
# imagePaths = glob.glob('./Images/60fps_office/*.png')
images = []

# loop over the image paths, load each one, and add them to our
# images to stich list
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)
    print(imagePath)

# initialize OpenCV's image sticher object and then perform the image
# stitching

if __name__ == "__main__":
    print("[INFO] stitching images...")
    stitcher = (
        cv2.createStitcher(cv2.Stitcher_PANORAMA)
        if imutils.is_cv3()
        else cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    )
    (status, stitched) = stitcher.stitch(images)

    # if the status is '0', then OpenCV successfully performed image
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

    # otherwise the stitching failed, likely due to not enough keypoints)
    # being detected
    else:
        print("[INFO] image stitching failed ({})".format(status))
