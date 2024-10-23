import cv2
import numpy as np
import glob
import datetime
import random
import time

start = time.time()
end = 0


class blender:

    def Laplacian_blending(self, img1, img2, mask, levels=4):

        G1 = img1.copy()
        G2 = img2.copy()
        GM = mask.copy()
        gp1 = [G1]
        gp2 = [G2]
        gpM = [GM]

        for i in range(levels - 1):
            G1 = cv2.pyrDown(G1)
            G2 = cv2.pyrDown(G2)
            GM = cv2.pyrDown(GM)
            gp1.append(np.float32(G1))
            gp2.append(np.float32(G2))
            gpM.append(np.float32(GM))

        lp1 = [gp1[levels - 1]]
        lp2 = [gp2[levels - 1]]
        gpMr = [gpM[levels - 1]]

        for i in range(levels - 1, 0, -1):
            L1 = np.subtract(gp1[i - 1], cv2.pyrUp(gp1[i]))
            L2 = np.subtract(gp2[i - 1], cv2.pyrUp(gp2[i]))
            lp1.append(L1)
            lp2.append(L2)
            gpMr.append(gpM[i - 1])

        LS = []
        for i, (l1, l2, gm) in enumerate(zip(lp1, lp2, gpMr)):
            ls = l1 * (gm) + l2 * (1 - gm)
            LS.append(ls)

        ls_ = LS[0]
        for i in range(1, levels):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[i])
        return ls_


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


def stitchCentral(detector, images):
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

        # findTransformed(i, homography)

        homographies[i] = homographies[i + 1] @ homography

    for i in range(central_idx + 1, len(images)):
        kp1, des1 = detector.detectAndCompute(images[i], None)
        kp2, des2 = detector.detectAndCompute(images[i - 1], None)

        bf = cv2.BFMatcher(matcher, crossCheck=True)
        matches = bf.match(des1, des2)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # findTransformed(i, homography)

        homographies[i] = homographies[i - 1] @ homography

    x_min, y_min, x_max, y_max = canvasSize(images, homographies)
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
        stitched = overlayImages(stitched, warped_image)

        # cv2.namedWindow("stitching", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("stitching", 1280, 720)
        # cv2.imshow("stitching", stitched)
        # cv2.waitKey(500)
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


def detectorKeypoint(detector, img, mask=None):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoint, descriptors = detector.detectAndCompute(gray, mask)
    return keypoint, descriptors


def BFMatch(stitchDescriptor, img_num, nmatches=500):

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(stitchDescriptor, arrdescriptors[img_num])
    matches = sorted(matches, key=lambda x: x.distance)

    # img_matches = cv2.drawMatches(
    #     img,
    #     stitchKeypoints,
    #     arrimage[img_num],
    #     arrkeypoints[img_num],
    #     matches[:nmatches],
    #     None,
    #     matchColor=(255, 255, 0),
    #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    # )
    # print(f"image {img_num} with image {img_num+1}: {len(matches)}")
    img_matches = 0
    return img_matches, matches[:nmatches]


def homMatrix(stitchKeypoints, img_num, match):
    srcKpts = np.float32([(stitchKeypoints)[m.queryIdx].pt for m in match]).reshape(
        -1, 1, 2
    )

    dtsKpts = np.float32(
        [(arrkeypoints[img_num])[m.trainIdx].pt for m in match]
    ).reshape(-1, 1, 2)

    Hom, mask = cv2.findHomography(dtsKpts, srcKpts, cv2.RANSAC, 10.0)

    inlier_matches = [match[i] for i in range(len(match)) if mask[i]]
    confidence = len(inlier_matches) / (1 + (0.1 * len(match)))
    img_matches = 0
    return img_matches, confidence, Hom


def find_best_match(current_image, excluded_images, matrix):
    best_match = None
    best_score = -1
    for img_idx, score in enumerate(matrix[current_image]):
        if img_idx not in excluded_images and score > best_score:
            best_match = img_idx
            best_score = score
    return best_match


def generateOrderedImages(matrix):
    current_image = random.choice(range(len(matrix)))
    # current_image = 0
    print(f"Initial center image: {current_image}")

    order = [current_image]

    while len(order) < len(matrix):
        left_best = find_best_match(order[0], order, matrix)
        right_best = find_best_match(order[-1], order, matrix)

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

# imagePaths = natsorted(list(glob.glob("../../Images/60fps_office/*")), reverse=False)
# imagePaths = list(glob.glob("../../Images/dumdumset/*"))
# imagePaths = natsorted(list(glob.glob("./st4/*")), reverse=False)
path = "./st4/*"
imagePaths = list(glob.glob(path))
print(f"Images Path: {path}")

arrimage = []
arrimgname, arrkeypoints, arrdescriptors = [], [], []
newOrderIdx, newImmageOrder = [], []
matxConf = []

heightth = []
widthth = []

sift = cv2.SIFT.create()  # cv.NORM_L2
orb = cv2.ORB.create()  # cv2.NORM_HAMMING
brisk = cv2.BRISK.create()  # cv.NORM_L2
akaze = cv2.AKAZE.create()  # cv.NORM_L2


if __name__ == "__main__":
    for image in imagePaths:
        readImage = cv2.imread(image)

        blurlevel = ip.detectBlur(readImage)
        if blurlevel < 100:
            continue

        readImage = ip.resizeImage(readImage)
        # readImage = ip.imagePreProcess(readImage)

        imageKeypoint, imageDescriptor = detectorKeypoint(brisk, readImage)

        arrimage.append(readImage)
        arrimgname.append(image)
        arrdescriptors.append(imageDescriptor)
        arrkeypoints.append(imageKeypoint)
        
        print(f"Extracting Images: ({len(arrimage)}/{len(imagePaths)})")
        
    print(f"Total Image In Folder: {len(imagePaths)}")
    print(f"Blur Images: {len(imagePaths) - len(arrimage)}")
    print(f"Total Images Now: {len(arrimage)}")
    # ========================================================================================
    for i in range(len(arrimage)):
        arrconfidence = []
        for j in range(len(arrimage)):

            imgMatch, matches = BFMatch(arrdescriptors[i], j, 500)
            imgMatch, confidence, homogMatrix = homMatrix(arrkeypoints[i], j, matches)

            if arrimgname[i] == arrimgname[j]:
                confidence = 0

            arrconfidence.append(int(confidence * 10))

        conf = max(arrconfidence)
        confIndex = arrconfidence.index(conf)
        matxConf.append(arrconfidence)

    newOrderIdx = generateOrderedImages(matxConf)
    print(newOrderIdx)
    # ========================================================================================

    for i in range(len(arrimage)):
        newImmageOrder.append(arrimage[newOrderIdx[i]])
        # cv2.imshow("new img", new_img_order[i])
        # cv2.waitKey(0)


    stitched_image = stitchCentral(brisk, newImmageOrder)

    cv2.imwrite(
        f"../../Results/{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}_stitched_final.png",
        stitched_image,
    )

    # ip = ImageProcessor(scale=0.5)
    # stitched_image = ip.resizeImage(stitched_image)
    # cv2.imshow("stitched_output", stitched_image)
    # cv2.waitKey(500)

    # print(heightth)
    print()
    # print(widthth)

    print("=== DONE ===")
    end = time.time()
    print(f"Time elapsed: {end-start}")
