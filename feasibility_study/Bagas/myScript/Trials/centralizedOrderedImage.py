import cv2
import numpy as np
import glob
import datetime
import random
import time
from natsort import natsorted

start = time.time()
end = 0


def laplacianBlending(img1, img2, levels=4):
    # Generate Gaussian pyramid for img1
    G1 = img1.copy()
    gp1 = [G1]
    for i in range(levels):
        G1 = cv2.pyrDown(G1)
        gp1.append(G1)

    # Generate Gaussian pyramid for img2
    G2 = img2.copy()
    gp2 = [G2]
    for i in range(levels):
        G2 = cv2.pyrDown(G2)
        gp2.append(G2)

    # Generate Laplacian pyramid for img1
    lp1 = [gp1[levels]]
    for i in range(levels, 0, -1):
        GE1 = cv2.pyrUp(gp1[i])
        # Resize GE1 to match gp1[i - 1]
        GE1 = cv2.resize(GE1, (gp1[i - 1].shape[1], gp1[i - 1].shape[0]))
        L1 = cv2.subtract(gp1[i - 1], GE1)
        lp1.append(L1)

    # Generate Laplacian pyramid for img2
    lp2 = [gp2[levels]]
    for i in range(levels, 0, -1):
        GE2 = cv2.pyrUp(gp2[i])
        # Resize GE2 to match gp2[i - 1]
        GE2 = cv2.resize(GE2, (gp2[i - 1].shape[1], gp2[i - 1].shape[0]))
        L2 = cv2.subtract(gp2[i - 1], GE2)
        lp2.append(L2)

    # Now blend the Laplacian pyramids
    LS = []
    for l1, l2 in zip(lp1, lp2):
        rows, cols, dpt = l1.shape
        ls = np.hstack((l1[:, : cols // 2], l2[:, cols // 2 :]))
        LS.append(ls)

    # Reconstruct the image
    blended_image = LS[0]
    for i in range(1, levels + 1):
        blended_image = cv2.pyrUp(blended_image)
        # Resize to match the current level
        blended_image = cv2.resize(blended_image, (LS[i].shape[1], LS[i].shape[0]))
        blended_image = cv2.add(blended_image, LS[i])

    return blended_image


def uniformBslend(img1, img2):
    # grayscale
    gray1 = np.mean(img1, axis=-1)
    gray2 = np.mean(img2, axis=-1)
    result = img1.astype(np.float64) + img2.astype(np.float64)

    g1, g2 = gray1 > 0, gray2 > 0
    g = g1 & g2
    mask = np.expand_dims(g * 0.5, axis=-1)
    mask = np.tile(mask, [1, 1, 3])
    mask[mask == 0] = 1
    result *= mask
    result = result.astype(np.uint8)

    return result


def findTransformed(i, homography):

    h, w, _ = arrimage[i].shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    transformed_corner = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography)

    minx = np.min(transformed_corner[:, 0, 0])
    maxx = np.max(transformed_corner[:, 0, 0])
    miny = np.min(transformed_corner[:, 0, 1])
    maxy = np.max(transformed_corner[:, 0, 1])

    print(f"pic {i} : {minx}, {maxx}, {miny}, {maxy}")
    print(f"pic {i} W: {maxx - minx} H: {maxy - miny}")
    print()
    heightth.append(int(maxy - miny))
    widthth.append(int(maxx - minx))
    # cv2.waitKey(500)


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

    def detectBlur(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurlevel = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blurlevel


class DetectMatchConfidence:
    def __init__(self):
        print()

    def detectorKeypoint(self, detector, img, mask=None):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoint, descriptors = detector.detectAndCompute(gray, mask)
        return keypoint, descriptors

    def BFMatch(self, stitchDescriptor, img_num, nmatches=500):

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(stitchDescriptor, arrdescriptors[img_num])
        matches = sorted(matches, key=lambda x: x.distance)

        return matches[:nmatches]

    def findConfidenceMatch(self, stitchKeypoints, img_num, match, allowedPixels=4):

        srcKpts = np.float32([(stitchKeypoints)[m.queryIdx].pt for m in match]).reshape(
            -1, 1, 2
        )

        dtsKpts = np.float32(
            [(arrkeypoints[img_num])[m.trainIdx].pt for m in match]
        ).reshape(-1, 1, 2)

        Hom, mask = cv2.findHomography(dtsKpts, srcKpts, cv2.RANSAC, allowedPixels)

        inlier_matches = [match[i] for i in range(len(match)) if mask[i]]
        confidence = len(inlier_matches) / (1 + (0.1 * len(match)))

        return confidence

    def BFMatchKNN(self, stitchDescriptor, img_num, k=2, ratio_thresh=0.75):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(stitchDescriptor, arrdescriptors[img_num], k=k)

        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

        return good_matches

    def knnConfidenceMatch(self, match, stitchKeypoints, img_num):
        srcKpts = np.float32([(stitchKeypoints)[m.queryIdx].pt for m in match]).reshape(
            -1, 1, 2
        )

        dtsKpts = np.float32(
            [(arrkeypoints[img_num])[m.trainIdx].pt for m in match]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(srcKpts, dtsKpts, cv2.RANSAC, 5.0)

        inlier_matches = [match[i] for i in range(len(match)) if mask[i]]
        confidence = len(inlier_matches) / (1 + (0.1 * len(match)))

        return confidence


def canvasSize(images, homographies):
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


def stitchCentral(detector, images, allowedPixels=4):
    matcher = cv2.NORM_HAMMING if detector == cv2.ORB.create() else cv2.NORM_L2

    central_idx = len(images) // 2

    homographies = [np.eye(3) for _ in range(len(images))]
    print("left")
    for i in range(central_idx - 1, -1, -1):
        kp1, des1 = detector.detectAndCompute(images[i], None)
        kp2, des2 = detector.detectAndCompute(images[i + 1], None)

        bf = cv2.BFMatcher(matcher, crossCheck=True)
        matches = bf.match(des1, des2)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, allowedPixels)

        # findTransformed(i, homography)

        homographies[i] = homographies[i + 1] @ homography

    print("right")
    for i in range(central_idx + 1, len(images)):
        kp1, des1 = detector.detectAndCompute(images[i], None)
        kp2, des2 = detector.detectAndCompute(images[i - 1], None)

        bf = cv2.BFMatcher(matcher, crossCheck=True)
        matches = bf.match(des1, des2)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, allowedPixels)

        # findTransformed(i, homography)

        homographies[i] = homographies[i - 1] @ homography

    print("canvas")
    x_min, y_min, x_max, y_max = canvasSize(images, homographies)
    canvas_width, canvas_height = x_max - x_min, y_max - y_min

    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    adjusted_homographies = [translation_matrix @ h for h in homographies]

    stitched = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    print("overlay")
    for i in range(len(images)):
        # print(f"image {i}")
        # cv2.namedWindow("stitching", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("stitching", 1280, 720)
        # cv2.imshow("stitching", stitched)
        # cv2.waitKey(1000)

        warped_image = cv2.warpPerspective(
            images[i],
            adjusted_homographies[i],
            (canvas_width, canvas_height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

        stitched = overlayImages(stitched, warped_image)
        # stitched = laplacianBlending(stitched, warped_image)

        # cv2.imwrite(
        #     f"./stRes/{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}_parts.png",
        #     warped_image,
        # )

    return stitched


def overlayImages(base_img, new_img):
    assert (
        base_img.shape == new_img.shape
    ), "Images must have the same shape for overlaying."

    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    mask = gray > 0

    base_img[mask] = new_img[mask]

    return base_img


def findBestMatch(current_image, excluded_images, matrix):
    best_match = None
    best_score = -1
    for img_idx, score in enumerate(matrix[current_image]):
        if img_idx not in excluded_images and score > best_score:
            best_match = img_idx
            best_score = score
    return best_match


def generateOrderedImages(matrix):
    current_image = random.choice(range(len(matrix)))

    print(f"Initial center image: {current_image}")

    order = [current_image]

    while len(order) < len(matrix):
        left_best = findBestMatch(order[0], order, matrix)
        right_best = findBestMatch(order[-1], order, matrix)

        if left_best is not None and right_best is not None:
            left_score = matrix[order[0]][left_best]
            right_score = matrix[order[-1]][right_best]

            if left_score > right_score:
                order.insert(0, left_best)
            else:
                order.append(right_best)
        elif left_best is not None:
            order.insert(0, left_best)
        elif right_best is not None:
            order.append(right_best)

        # print(f"Current order: {order}")

    return order


imagescale = 1
ip = ImageProcessor(imagescale)
dmc = DetectMatchConfidence()

path = "./st1/*"
imagePaths = natsorted(list(glob.glob(path)))
print(f"Images Path: {path}")

arrimage = []
arrimgname, arrkeypoints, arrdescriptors = [], [], []
newOrderIdx = []
matxConf = []

heightth = []
widthth = []

sift = cv2.SIFT.create()  # cv.NORM_L2
orb = cv2.ORB.create()  # cv2.NORM_HAMMING
brisk = cv2.BRISK.create()  # cv.NORM_L2
akaze = cv2.AKAZE.create()  # cv.NORM_L2


method = "bf"

if __name__ == "__main__":
    for image in imagePaths:
        readImage = cv2.imread(image)
        readImage = ip.resizeImage(readImage)
        readImage = ip.imagePreProcess(readImage)

        blurlevel = ip.detectBlur(readImage)
        # print("blur level", blurlevel)
        if blurlevel < 100:
            continue

        imageKeypoint, imageDescriptor = dmc.detectorKeypoint(brisk, readImage)

        arrimage.append(readImage)
        arrimgname.append(image)
        arrdescriptors.append(imageDescriptor)
        arrkeypoints.append(imageKeypoint)

        print(f"Extracting Images: ({len(arrimage)}/{len(imagePaths)})")

    print(f"Total Image In Folder: {len(imagePaths)}")
    print(f"Blur Images: {len(imagePaths) - len(arrimage)}")
    print(f"Total Images Now: {len(arrimage)}")

    arrdone = []
    print(type(arrkeypoints[0][0]))

    for i in range(len(arrimage)):
        arrconfidence = []
        for j in range(len(arrimage)):

            """
            # nmatches a.k.a number of matches, the higher the value will make the stitching and pairing more accurate
            # also note that the bigger the number of nmatches the longer the time will be to be calculated.
            # reccomended is 2000 matches for most images.
            """
            if j in arrdone:
                confidence = 0
                arrconfidence.append(int(confidence))
                continue

            if arrimgname[i] == arrimgname[j]:
                confidence = 0
                arrconfidence.append(int(confidence))
                continue

            # KNN
            if method == "knn":
                matches = dmc.BFMatchKNN(arrdescriptors[i], j)
                confidence = dmc.knnConfidenceMatch(matches, arrkeypoints[i], j)

            # Normal
            elif method == "bf":
                matches = dmc.BFMatch(arrdescriptors[i], j, nmatches=2000)
                confidence = dmc.findConfidenceMatch(arrkeypoints[i], j, matches)

            print(f"{i+1} :: {j+1} == {confidence}")

            arrconfidence.append(int(confidence))

        arrdone.append(i)
        # print(f"arrdone: {arrdone}")

        # if not arrconfidence:
        #     break

        conf = max(arrconfidence)
        # print(f"conf: {conf}")
        confIndex = arrconfidence.index(conf)
        matxConf.append(arrconfidence)
        # print(f"matxConf: {matxConf}")

    newOrderIdx = generateOrderedImages(matxConf)
    print(newOrderIdx)

    print("copying order")
    newImageOrder = [arrimage[i] for i in newOrderIdx]

    """
    # allowedPixels are the distance of error between two points of matches. 
    # for bigger images, it can use to near 0 numbers
    # for normal images, it can use numbers ranging from 10 ~ 20
    # the general allowedPixels used for every pictures default at 4
    """

    print("central")
    stitched_image = stitchCentral(brisk, newImageOrder, allowedPixels=10)

    cv2.imwrite(
        f"../../Results/{datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}_st_{method}.png",
        stitched_image,
    )

    # print(heightth)
    print()
    # print(widthth)

    print("=== DONE ===")
    end = time.time()
    print(f"Time elapsed: {end-start}")

    ip = ImageProcessor(scale=0.5)
    stitched_image = ip.resizeImage(stitched_image)
    cv2.namedWindow("stitched_output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("stitched_output", 1280, 720)
    cv2.imshow("stitched_output", stitched_image)
    cv2.waitKey(0)
