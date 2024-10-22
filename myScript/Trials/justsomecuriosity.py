import cv2
import numpy as np
from natsort import natsorted
import glob
import datetime


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


def calculate_canvas_size(images, homographies):
    height, width = images[0].shape[:2]

    all_corners = []

    for i, homography in enumerate(homographies):
        corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        ).reshape(-1, 1, 2)

        transformed_corners = cv2.perspectiveTransform(corners, homography)
        all_corners.append(transformed_corners)

    all_corners = np.concatenate(all_corners, axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

    return x_min, y_min, x_max, y_max


def stitch_images_central(detector, images):
    matcher = cv2.NORM_HAMMING if detector != cv2.SIFT.create() else cv2.NORM_L1

    central_idx = len(images) // 2

    homographies = [np.eye(3) for _ in range(len(images))]

    for i in range(central_idx - 1, -1, -1):
        kp1, des1 = detector.detectAndCompute(images[i], None)
        kp2, des2 = detector.detectAndCompute(images[i + 1], None)

        bf = cv2.BFMatcher(matcher, crossCheck=True)
        matches = bf.match(des1, des2)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homographies[i] = homographies[i + 1] @ homography

    for i in range(central_idx + 1, len(images)):
        kp1, des1 = detector.detectAndCompute(images[i], None)
        kp2, des2 = detector.detectAndCompute(images[i - 1], None)

        bf = cv2.BFMatcher(matcher, crossCheck=True)
        matches = bf.match(des1, des2)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homographies[i] = homographies[i - 1] @ homography

    x_min, y_min, x_max, y_max = calculate_canvas_size(images, homographies)
    canvas_width, canvas_height = x_max - x_min, y_max - y_min

    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    adjusted_homographies = [translation_matrix @ h for h in homographies]

    stitched = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for i in range(len(images)):
        warped_image = cv2.warpPerspective(
            images[i],
            adjusted_homographies[i],
            (canvas_width, canvas_height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        stitched = overlay_images(stitched, warped_image)

        cv2.namedWindow("stitching", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("stitching", 1280, 720)
        cv2.imshow("stitching", stitched)
        cv2.waitKey(500)

    return stitched


def overlay_images(base_img, new_img):
    assert (
        base_img.shape == new_img.shape
    ), "Images must have the same shape for overlaying."

    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    mask = gray > 0
    base_img[mask] = new_img[mask]

    return base_img


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def detectBlur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurlevel = variance_of_laplacian(gray)
    return blurlevel


imagescale = 1
ip = ImageProcessor(imagescale)

imagePaths = natsorted(list(glob.glob("../../Images/60fps_office/*")), reverse=False)
images = []

sift = cv2.SIFT.create()  # cv.NORM_L2
orb = cv2.ORB.create()  # cv2.NORM_HAMMING
brisk = cv2.BRISK.create()  # cv.NORM_L2
akaze = cv2.AKAZE.create()  # cv.NORM_L2

if __name__ == "__main__":
    for image in imagePaths:
        readImage = cv2.imread(image)
        blurlevel = detectBlur(readImage)
        if blurlevel < 200:
            continue
        readImage = ip.resizeImage(readImage)
        readImage = ip.imagePreProcess(readImage)

        images.append(readImage)

    stitched_image = stitch_images_central(sift, images)
    ip = ImageProcessor(0.5)
    stitched_image = ip.resizeImage(stitched_image)
    cv2.imshow("stitched_output", stitched_image)
    cv2.waitKey(0)
    cv2.imwrite(
        f"../../Results/{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}_stitched_final.png",
        stitched_image,
    )

    print("=== DONE ===")
