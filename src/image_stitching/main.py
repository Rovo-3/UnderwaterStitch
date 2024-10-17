from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from datetime import datetime
import os

def dynamic_white_balance(image):
    wb = cv2.xphoto.createSimpleWB()
    corrected_image = wb.balanceWhite(image)
    # cv2.imwrite("ImprovedSIFT_correctedWB.jpg", corrected_image)
    # showimg("correctedWB", corrected_image)
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
    # clahe_img = cv2.imwrite("ImprovedSIFT_clahe.jpg",output)
    # showimg("output", output)
    return output

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str,
	help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str,
	help="path to the output image")
ap.add_argument("-c", "--crop", type=int, default=1,
	help="whether to crop out largest rectangular region")
args = vars(ap.parse_args())

folder_name = "VID1"
output_path = "./output/" + folder_name + "/"
input_path = "./images/generated_frames/" + folder_name

try:
    os.makedirs(output_path)
    print(f"Directory '{output_path}' created")
except FileExistsError:
    print(f"Directory '{output_path}' already exists")

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
# imagePaths = sorted(list(paths.list_images("./images/set13")))

imagePaths = sorted(list(paths.list_images(input_path)), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
print(imagePaths)
images = []

# loop over the image paths, load each one, and add them to our
# images to stich list
for imagePath in imagePaths:
    image=cv2.imread(imagePath)
    # image=dynamic_white_balance(image)
    image=apply_CLAHE(image)
    # image=cv2.resize(image,(0,0),fx=1, fy=1)
    # image = cv2.undistort (image, camera_matrix, dist_coeffs)
    images.append(image)

print("[INFO] stitching images...")
stitcher = cv2.Stitcher.create(cv2.STITCHER_SCANS)
(status, stitched) = stitcher.stitch(images)
print("done stitching")

# if the status is '0', then OpenCV successfully performed image
if status == 0:
    now = datetime.now()
    date = now.strftime('%y%m%d_%H%M%S')
    filename=output_path+str(date)+".jpg"
    print(filename)

    print("status:", status)
    cv2.imwrite(filename, stitched)
    # display the output stitched image to our screen
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)
# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
    print("[INFO] image stitching failed ({})".format(status))



# cropping is not yet required.
    # if args["crop"] > 0:
    #     # create a 10 pixel border surrounding the stitched image
    #     print("[INFO] cropping...")
    #     stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
    #         cv2.BORDER_CONSTANT, (0, 0, 0))
    #     # convert the stitched image to grayscale and threshold it
    #     # such that all pixels greater than zero are set to 255
    #     # (foreground) while all others remain 0 (background)
    #     gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    #     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    #             # find all external contours in the threshold image then find
    #     # the *largest* contour which will be the contour/outline of
    #     # the stitched image
    #     cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    #         cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = imutils.grab_contours(cnts)
    #     c = max(cnts, key=cv2.contourArea)
    #     # allocate memory for the mask which will contain the
    #     # rectangular bounding box of the stitched image region
    #     mask = np.zeros(thresh.shape, dtype="uint8")
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    #             # create two copies of the mask: one to serve as our actual
    #     # minimum rectangular region and another to serve as a counter
    #     # for how many pixels need to be removed to form the minimum
    #     # rectangular region
    #     minRect = mask.copy()
    #     sub = mask.copy()
    #     # keep looping until there are no non-zero pixels left in the
    #     # subtracted image
    #     while cv2.countNonZero(sub) > 0:
    #         # erode the minimum rectangular mask and then subtract
    #         # the thresholded image from the minimum rectangular mask
    #         # so we can count if there are any non-zero pixels left
    #         minRect = cv2.erode(minRect, None)
    #         sub = cv2.subtract(minRect, thresh)
    #     # find contours in the minimum rectangular mask and then
    #     # extract the bounding box (x, y)-coordinates
    #     cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = imutils.grab_contours(cnts)
    #     c = max(cnts, key=cv2.contourArea)
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     # use the bounding box coordinates to extract the our final
    #     # stitched image
    #     stitched = stitched[y:y + h, x:x + w]
    #     cv2.imwrite(filename, stitched)
    #     # display the output stitched image to our screen
    #     cv2.imshow("Stitched", stitched)
    #     cv2.waitKey(0)
    # else: