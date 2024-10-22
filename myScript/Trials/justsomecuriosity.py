import cv2
import glob
from natsort import natsorted
import numpy as np


def pyforceClose():
    import sys

    print("Closed, Thank You")

    sys.exit()


def wbOpenCV(img):
    wb = cv2.xphoto.createSimpleWB()
    wb_img = wb.balanceWhite(img)
    return wb_img


def chanelClahe(channel):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    channel_clahe = clahe.apply(channel)
    return channel_clahe


def imagePreProcess(img, sigma, kernel, blur=False):
    white_balanced_img = wbOpenCV(img)
    img = white_balanced_img

    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    l_channel_clahe = chanelClahe(l_channel)
    merged_lab = cv2.merge((l_channel_clahe, a_channel, b_channel))

    final_img_lab = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    if blur:
        final_img_lab = cv2.GaussianBlur(final_img_lab, kernel, sigma)
    return final_img_lab


def resizeImage(img, scale):
    h, w, _ = img.shape
    h = h * scale
    w = w * scale
    img = cv2.resize(img, (int(w), int(h)))
    return img


def detectorKeypoint(detector, img, mask=None):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoint, descriptors = detector.detectAndCompute(gray, mask)
    matcher = cv2.NORM_HAMMING if detector != cv2.SIFT.create() else cv2.NORM_L1
    return keypoint, descriptors, matcher


def BFMatch(matcher, img, stitchKeypoints, stitchDescriptor, img_num, nmatches=500):

    bf = cv2.BFMatcher(matcher, crossCheck=True)
    matches = bf.match(stitchDescriptor, arrdescriptors[img_num + 1])
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(
        img,
        stitchKeypoints,
        arrimage[img_num + 1],
        arrkeypoints[img_num + 1],
        matches[:nmatches],
        None,
        matchColor=(255, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    # print(f"image {img_num} with image {img_num+1}: {len(matches)}")

    return img_matches, matches[:nmatches]


def BFMatchKNN(img_num):
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        img_matches = bf.knnMatch(
            arrdescriptors[img_num], arrdescriptors[img_num + 1], k=2
        )

        for i, j in img_matches:
            if i.distance < 0.75 * j.distance:
                goodkeypoints.append([i])

        # print(f"image {img_num} with image {img_num+1}: {len(goodkeypoints[i])}")

        img_matches = cv2.drawMatchesKnn(
            arrimage[img_num],
            arrkeypoints[img_num],
            arrimage[img_num + 1],
            arrkeypoints[img_num + 1],
            goodkeypoints,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return img_matches
    except cv2.error as e:
        print(e)


def homMatrix(stitchKeypoints, img_num, match):
    srcKpts = np.float32([(stitchKeypoints)[m.queryIdx].pt for m in match]).reshape(
        -1, 1, 2
    )

    dtsKpts = np.float32(
        [(arrkeypoints[img_num + 1])[m.trainIdx].pt for m in match]
    ).reshape(-1, 1, 2)

    Hom, mask = cv2.findHomography(dtsKpts, srcKpts, cv2.RANSAC, 10.0)
    matchesMask = mask.ravel().tolist()

    inlier_matches = [match[i] for i in range(len(match)) if mask[i]]
    print("Inlier Match: ", len(inlier_matches))
    # confidence = len(inlier_matches) / (8 + (0.3 * len(match)))
    confidence = len(inlier_matches) / (1 + (0.1 * len(match)))
    print("Matches: ", len(match))
    print("Confidence: ", confidence)

    return confidence, Hom


def commandStitch(img, nextimg, HomMatx):

    h1, w1, _ = img.shape
    h2, w2, _ = nextimg.shape

    img2Warped = cv2.warpPerspective(nextimg, HomMatx, (w1 + w2, max(h1, h2)))

    stitchedimg = np.copy(img2Warped)

    stitchedimg[0:h1, 0:w1] = img

    print(f"h1: {h1}, w1: {w1} <-- h2: {h2}, w2: {w2}")

    return stitchedimg


def trimRightSide(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    ret, thresh = cv2.threshold(gray, 1, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    max_area = -1
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    approx = cv2.approxPolyDP(best_cnt, 0.01 * cv2.arcLength(best_cnt, True), True)

    distances = np.linalg.norm(approx[:, 0, :], axis=1)
    far = approx[distances.argmax()][0]

    ymax = approx[:, :, 1].max()
    xmax = approx[:, :, 0].max()

    x = min(far[0], xmax)
    y = min(far[1], ymax)

    img2 = img[:y, :x].copy()
    return img2


def trimBlackArea(stitched_img):
    # Convert to grayscale
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)

    # Find all non-black pixel coordinates
    non_black_pixels = np.nonzero(gray)

    # If no non-black pixels are found, return the original image
    if len(non_black_pixels[0]) == 0:
        print("returned original")
        return stitched_img

    # Get the bounding box for non-black pixels
    min_y, max_y = np.min(non_black_pixels[0]), np.max(non_black_pixels[0])
    min_x, max_x = np.min(non_black_pixels[1]), np.max(non_black_pixels[1])

    # Crop the image to this bounding box
    cropped_img = stitched_img[min_y : max_y + 1, min_x : max_x + 1]
    print(f"y1:{min_y}, y2:{max_y}, x1:{min_x}, x2{max_x}")
    cropped_img = trimRightSide(cropped_img)
    return cropped_img


sift = cv2.SIFT.create()
orb = cv2.ORB.create()
brisk = cv2.BRISK.create()
akaze = cv2.AKAZE.create()

# imagePaths = natsorted(list(glob.glob("../../Images/2ndfloor/*")), reverse=False)
imagePaths = natsorted(list(glob.glob("./st1/*")), reverse=False)

arrimage, arrkeypoints, arrdescriptors = [], [], []

goodkeypoints = []
notStitched = []

screensize = 0.5

if __name__ == "__main__":

    for imgs in imagePaths:
        readImage = cv2.imread(imgs)
        readImage = resizeImage(readImage, 1)
        readImage = imagePreProcess(readImage, 0, (1, 1), False)

        keypoint, descr, _ = detectorKeypoint(brisk, readImage)
        print(f"Load Images: ({len(arrimage)+1}/{len(imagePaths)})")

        arrimage.append(readImage)
        arrdescriptors.append(descr)
        arrkeypoints.append(keypoint)

    stitchedimg = arrimage[0]

    for imagenumber in range(len(arrimage)):
        if imagenumber == len(arrimage) - 1:
            break
        stitchKeypoints, stitchDescriptor, matcher = detectorKeypoint(brisk, stitchedimg)

        img_matches, matchess = BFMatch(
            matcher, stitchedimg, stitchKeypoints, stitchDescriptor, imagenumber, 2000
        )
        print(f"Image {imagenumber+1} with {imagenumber+2}")

        confidence, HomogMatx = homMatrix(stitchKeypoints, imagenumber, matchess)
        if confidence >= 0.04:
            nextimg = arrimage[imagenumber + 1]
            stitchedimg = commandStitch(stitchedimg, nextimg, HomogMatx)
            stitchedimg = trimBlackArea(stitchedimg)

            h, w, _ = stitchedimg.shape

            cv2.namedWindow("stitched", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("stitched", int(w*screensize), int(h*screensize))
            cv2.imshow("stitched", stitchedimg)
            cv2.waitKey(500)

        else:
            cv2.destroyAllWindows()
            print("Not Stitched")
            notStitched.append(imagenumber)
        print("===========")

    print("=== FINISH ===")
    print("Number of images left: ", len(notStitched))
    print("Not Stitched Image", notStitched)

    cv2.namedWindow("Final Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Final Image", 1280, 720)
    cv2.imshow("Final Image", stitchedimg)
    cv2.imwrite(f"./stRes/st1resa0000a{imagenumber+1}.jpg", stitchedimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
