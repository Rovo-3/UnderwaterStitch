from stitching import Stitcher
import numpy as np
import imutils
import cv2
import datetime
import time
from natsort import natsorted
import glob

start = time.time()
end = 0

outpath = (
    "../Results/"
    + datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    + "_stitched.png"
)


def pyforceClose():
    import sys

    sys.exit()


class ImageProcessor:
    def __init__(self, scale=1.0):
        self.scale = scale

    def wb_opencv(self, img):
        """Performs white balance on the image using OpenCV."""
        wb = cv2.xphoto.createSimpleWB()
        wb_img = wb.balanceWhite(img)
        return wb_img

    def chanelClahe(self, channel):
        """Applies CLAHE to a single image channel."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        channel_clahe = clahe.apply(channel)
        return channel_clahe

    def imagePreProcess(self, imagetobeprocessed):
        """Applies white balance and CLAHE preprocessing to the image."""
        # White balance
        imagetobeprocessed = self.wb_opencv(imagetobeprocessed)

        # Convert to LAB color space
        lab_image = cv2.cvtColor(imagetobeprocessed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply CLAHE to the L channel
        l_channel_clahe = self.chanelClahe(l_channel)

        # Merge the channels and convert back to BGR
        merged_lab = cv2.merge((l_channel_clahe, a_channel, b_channel))
        final_img_lab = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        return final_img_lab

    def resizeImage(self, img):
        """Resizes the image by the given scale factor."""
        h, w, _ = img.shape
        h = int(h * self.scale)
        w = int(w * self.scale)
        resized_img = cv2.resize(img, (w, h))
        return resized_img


print("[INFO] loading images...")
# imagePaths = natsorted(list(glob.glob("../Images/seaTrial30pics/*")))
imagePaths = natsorted(list(glob.glob("../myScript/Trials/st2/*")))
# imagePaths = natsorted(list(glob.glob("./dumdum/*")))

totalImages = len(imagePaths)

images = []

if __name__ == "__main__":

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        ip = ImageProcessor(1)

        processedimage = ip.imagePreProcess(image)
        # processedimage = wb_opencv(image)
        # cv2.imshow("images",processedimage)
        # cv2.waitKey(0)
        images.append(processedimage)
        print(imagePath)

    print("[INFO] stitching images...")
    # print(images)
    # scan = cv2.Stitcher_SCANS
    # pano = cv2.Stitcher_PANORAMA
    # (status, stitched) = cv2.Stitcher.create(scan).stitch(images)

    settings = {
        "detector": "sift",
        "confidence_threshold": 0.5
        # "blender_type": "feather",
        # "crop": False,
        # "no-crop": "True",
        # "wave_correct_kind": "auto"
        # "try_use_gpu": True
    }
    stitcher = Stitcher(**settings)
    stitched = stitcher.stitch(images)

    # if status == 0:
    print("[INFO] save stitching...")
    cv2.imwrite(outpath, stitched)

    print("[INFO] displayed stitching...")
    cv2.namedWindow("Stitched", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stitched", 1280, 720)
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(500)

    # else:
    #     print("[INFO] image stitching failed ({})".format(status))

print("=== DONE ===")
end = time.time()
print(f"Time elapsed: {end-start}")
