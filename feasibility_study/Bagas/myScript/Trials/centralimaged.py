import cv2
import numpy as np
from natsort import natsorted
import glob


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
        imagetobeprocessed = self.wb_opencv(imagetobeprocessed)

        lab_image = cv2.cvtColor(imagetobeprocessed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        l_channel_clahe = self.chanelClahe(l_channel)

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


def stitch_images_central(detector, images):
    # BF Matcher
    matcher = cv2.NORM_HAMMING if detector != cv2.SIFT.create() else cv2.NORM_L1
        
    # Select the central image as the anchor
    central_idx = len(images) // 2
    central_image = images[central_idx]

    # Initialize the stitched image as the central image
    stitched = np.zeros_like(central_image)  # Blank canvas for stitching

    # Create a list of homographies for each image (initialize to identity matrix)
    homographies = [np.eye(3) for _ in range(len(images))]

    # Compute homographies for images to the left of the central image
    for i in range(central_idx - 1, -1, -1):
        kp1, des1 = detector.detectAndCompute(images[i], None)
        kp2, des2 = detector.detectAndCompute(images[i + 1], None)

        # Match features using BFMatcher with L2 norm
        
        bf = cv2.BFMatcher(matcher, crossCheck=True)
        matches = bf.match(des1, des2)

        # Extract the matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography using RANSAC
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homographies[i] = homographies[i + 1] @ homography

    # Compute homographies for images to the right of the central image
    for i in range(central_idx + 1, len(images)):
        kp1, des1 = detector.detectAndCompute(images[i], None)
        kp2, des2 = detector.detectAndCompute(images[i - 1], None)

        # Match features using BFMatcher with L2 norm
        bf = cv2.BFMatcher(matcher, crossCheck=True)
        matches = bf.match(des1, des2)

        # Extract the matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography using RANSAC
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homographies[i] = homographies[i - 1] @ homography

    # Warp each image to the central reference frame and overlay them
    for i in range(len(images)):
        warped_image = cv2.warpPerspective(
            images[i], homographies[i], (central_image.shape[1], central_image.shape[0])
        )  # Resize to central image size
        stitched = overlay_images(stitched, warped_image)
        cv2.namedWindow("stitching", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("stitching", 1280, 720)
        cv2.imshow("stitching", stitched)
        cv2.waitKey(500)
        

    return stitched


def overlay_images(base_img, new_img):
    # Ensure the images are the same size
    assert (
        base_img.shape == new_img.shape
    ), "Images must have the same shape for overlaying."

    # Use the new image where it is not zero (non-transparent)
    mask = new_img > 0  # Create a mask for non-zero pixels
    base_img[mask] = new_img[mask]  # Overlay non-zero pixels onto the base image

    return base_img


imagescale = 1
ip = ImageProcessor(imagescale)

imagePaths = natsorted(list(glob.glob("./st1/*")))
# imagePaths = natsorted(list(glob.glob("../../Images/seaTrial30pics/*")))
imagePaths = natsorted(list(glob.glob("../../Images/60fps_office/*")))

images = []


# Initialize the SIFT detector
sift = cv2.SIFT.create() # cv.NORM_L2
orb = cv2.ORB.create() # cv2.NORM_HAMMING
brisk = cv2.BRISK.create() # cv.NORM_L2
akaze = cv2.AKAZE.create() # cv.NORM_L2

if __name__ == "__main__":
    for image in imagePaths:
        readImage = cv2.imread(image)
        readImage = ip.resizeImage(readImage)
        readImage = ip.imagePreProcess(readImage)

        images.append(readImage)

    stitched_image = stitch_images_central(brisk, images)
    ip = ImageProcessor(0.5)
    stitched_image = ip.resizeImage(stitched_image)
    cv2.imshow("stitched_output", stitched_image)
    cv2.waitKey(0)
    cv2.imwrite("stitched_output.jpg", stitched_image)

    print("=== DONE ===")

# [[0, 204, 74, 233], 
#  [204, 0, 306, 42], 
#  [74, 306, 0, 0], 
#  [233, 42, 0, 0]]
