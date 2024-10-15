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


def imagePreProcess(imagetobeprocessed, sigma, kernel, blur=False):
    white_balanced_img = wbOpenCV(imagetobeprocessed)
    imagetobeprocessed = white_balanced_img

    lab_image = cv2.cvtColor(imagetobeprocessed, cv2.COLOR_BGR2LAB)
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


def detectorKeypoint(detector, gray_img):
    keypoint, descriptors = detector.detectAndCompute(gray_img, None)
    return keypoint, descriptors


def BFMatch(img_num, nmatches=500):

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(arrdescriptors[img_num], arrdescriptors[img_num + 1])
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(
        arrimage[img_num],
        arrkeypoints[img_num],
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


def homMatrix(img_num, match):
    srcKpts = np.float32(
        [(arrkeypoints[img_num])[m.queryIdx].pt for m in match]
    ).reshape(-1, 1, 2)

    dtsKpts = np.float32(
        [(arrkeypoints[img_num + 1])[m.trainIdx].pt for m in match]
    ).reshape(-1, 1, 2)

    Hom, mask = cv2.findHomography(dtsKpts, srcKpts, cv2.RANSAC, 5.0)

    # Try to apply mask ==================================================================
    matchesMask = mask.ravel().tolist()

    draw_params = dict(
        matchColor=(0, 255, 0),  # Green color for inliers
        singlePointColor=(255, 0, 0),  # Red for keypoints
        matchesMask=matchesMask,  # Mask for matches
        flags=2,
    )

    img_matches = cv2.drawMatches(
        arrimage[img_num],
        arrkeypoints[img_num],
        arrimage[img_num + 1],
        arrkeypoints[img_num + 1],
        match,
        None,
        **draw_params,
    )

    # Try using the inlier =====================================
    img_matches = resizeImage(img_matches, 0.4)
    cv2.imshow("Inliers", img_matches)
    cv2.waitKey(500)

    inlier_matches = [match[i] for i in range(len(match)) if mask[i]]
    print("Inlier Match: ", len(inlier_matches))
    confidence = len(inlier_matches) / (8 + (0.3 * len(match)))
    print("Confidence: ", confidence)

    # inlier_src_pts = np.float32(
    #     [(arrkeypoints[img_num])[m.queryIdx].pt for m in inlier_matches]
    # ).reshape(-1, 2)
    # inlier_dst_pts = np.float32(
    #     [(arrkeypoints[img_num + 1])[m.trainIdx].pt for m in inlier_matches]
    # ).reshape(-1, 2)

    # Hom2, mask2 = cv2.findHomography(inlier_src_pts, inlier_dst_pts, cv2.RANSAC, 5.0)
    # Try using the inlier =====================================

    # Try to apply mask ==================================================================

    return confidence, Hom


def commandStitch(img_num, HomMatx):

    h1, w1, _ = arrimage[img_num].shape
    h2, w2, _ = arrimage[img_num + 1].shape

    img2Warped = cv2.warpPerspective(
        arrimage[img_num + 1], HomMatx, (w1 + w2, max(h1, h2))
    )

    stitchedimg = np.copy(img2Warped)

    alpha = 0.5
    blendedRegion = cv2.addWeighted(
        arrimage[img_num], alpha, stitchedimg[0:h1, 0:w1], 1 - alpha, 0
    )
    stitchedimg[0:h1, 0:w1] = blendedRegion
    stitchedimg = resizeImage(stitchedimg, 0.4)

    return stitchedimg


def stitched2(stitchedimg, nextimg, HomMatx):

    h1, w1, _ = stitchedimg.shape
    h2, w2, _ = nextimg.shape

    img2Warped = cv2.warpPerspective(nextimg, HomMatx, (w1 + w2, max(h1, h2)))

    stitchedimg = np.copy(img2Warped)

    alpha = 0.5
    blendedRegion = cv2.addWeighted(
        stitchedimg, alpha, stitchedimg[0:h1, 0:w1], 1 - alpha, 0
    )
    stitchedimg[0:h1, 0:w1] = blendedRegion
    stitchedimg = resizeImage(stitchedimg, 0.4)

    return stitchedimg


sift = cv2.SIFT.create()
orb = cv2.ORB.create()
brisk = cv2.BRISK.create()
akaze = cv2.AKAZE.create()

# imagePaths = natsorted(list(glob.glob("../../Images/2ndfloor/*")), reverse=False)
imagePaths = natsorted(list(glob.glob("../../Images/seaTrial30pics/*")), reverse=True)
# imagePaths = natsorted(list(glob.glob("../dumdum/*")))

arrimage = []
arrkeypoints = []
arrdescriptors = []
goodkeypoints = []
test = []

if __name__ == "__main__":

    for imgs in imagePaths:
        readImage = cv2.imread(imgs)
        readImage = resizeImage(readImage, 1)
        readImage = imagePreProcess(readImage, 0, (1, 1), False)

        cvtColorImg = cv2.cvtColor(readImage, cv2.COLOR_BGR2GRAY)
        keypoint, descr = detectorKeypoint(brisk, cvtColorImg)
        print(f"Images: ({len(arrimage)+1}/{len(imagePaths)})")

        drawnKeypoints = cv2.drawKeypoints(
            readImage, keypoint, None, color=(255, 255, 0)
        )
        arrimage.append(readImage)
        arrdescriptors.append(descr)
        arrkeypoints.append(keypoint)

    for imagenumber in range(len(arrimage)):
        if imagenumber == len(arrimage) - 1:
            break

        # Revise here to be using mask ==================
        img_matches, matchess = BFMatch(imagenumber, 2000)
        print(f"Image {imagenumber+1} with {imagenumber+2}")

        confidence, HomogMatx = homMatrix(
            imagenumber, matchess
        )  # <--- Revise the stitching using the masked inliners, take out any pictures with low inlier, then try stitching.
        # can also try to use the KNN matcher
        # Revise here to be using mask ==================
        if confidence >= 0.051:
            print("Stitched")
            stitchedimg = commandStitch(imagenumber, HomogMatx)

            cv2.putText(
                stitchedimg,
                "Confidence: {:.3f}".format(confidence),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.imshow("stitched", stitchedimg)
            cv2.waitKey(500)
        else:
            print("Not Stitched")
            test.append(imagenumber + 1)
        # cv2.imwrite(f"./res{imagenumber}.jpg", stitchedimg)
        print("===========")

    print("=== FINISH ===")
    print("not stitched", test)
    cv2.destroyAllWindows()

# ======================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

# def stitched2(stitchedimg, nextimg, HomMatx):

#     h1, w1, _ = stitchedimg.shape
#     h2, w2, _ = nextimg.shape

#     img2Warped = cv2.warpPerspective(nextimg, HomMatx, (w1 + w2, max(h1, h2)))

#     stitchedimg = np.copy(img2Warped)

#     alpha = 0.5
#     blendedRegion = cv2.addWeighted(
#         stitchedimg, alpha, stitchedimg[0:h1, 0:w1], 1 - alpha, 0
#     )
#     stitchedimg[0:h1, 0:w1] = blendedRegion
#     stitchedimg = resizeImage(stitchedimg, 0.4)

#     return stitchedimg

# stitchedimg = arrimage[0]
# stitchedimg = stitched2(stitchedimg, arrimage[i], HomogMatx)


# def commandStitch(stitchedimg, next_img, HomMatx):
#     h1, w1, _ = stitchedimg.shape
#     h2, w2, _ = next_img.shape

#     # Warp the next image using homography
#     img2Warped = cv2.warpPerspective(next_img, HomMatx, (w1 + w2, max(h1, h2)))

#     # Create a copy for blending
#     stitchedCanvas = np.copy(img2Warped)

#     # Blend overlapping regions using weighted sum
#     alpha = 0.5
#     blendedRegion = cv2.addWeighted(stitchedimg, alpha, stitchedCanvas[0:h1, 0:w1], 1 - alpha, 0)
#     stitchedCanvas[0:h1, 0:w1] = blendedRegion

#     # Optionally resize for display or output purposes
#     stitchedCanvas = resizeImage(stitchedCanvas, 0.4)

#     return stitchedCanvas