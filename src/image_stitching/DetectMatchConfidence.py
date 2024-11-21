import cv2
import numpy as np


class DetectMatchConfidence:
    def __init__(self):
        print()

    def detectorKeypoint(self, detector, img, mask=None):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoint, descriptors = detector.detectAndCompute(gray, mask)
        return keypoint, descriptors

    def BFMatch(self, stitchDescriptor, img_num, arrdescriptors, nmatches=500):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(stitchDescriptor, arrdescriptors[img_num])
        matches = sorted(matches, key=lambda x: x.distance)

        return matches[:nmatches]

    def BFMatchKNN(
        self, stitchDescriptor, img_num, arrdescriptors, k=2, ratio_thresh=0.75
    ):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(stitchDescriptor, arrdescriptors[img_num], k=k)

        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

        return good_matches

    def findConfidenceMatch(
        self, match, stitchKeypoints, img_num, arrkeypoints, allowedPixels=4
    ):

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

    def knnConfidenceMatch(
        self, match, stitchKeypoints, img_num, arrkeypoints, allowedPixels=10
    ):
        srcKpts = np.float32([(stitchKeypoints)[m.queryIdx].pt for m in match]).reshape(
            -1, 1, 2
        )

        dtsKpts = np.float32(
            [(arrkeypoints[img_num])[m.trainIdx].pt for m in match]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(srcKpts, dtsKpts, cv2.RANSAC, allowedPixels)

        inlier_matches = [match[i] for i in range(len(match)) if mask[i]]
        confidence = len(inlier_matches) / (1 + (0.1 * len(match)))

        return confidence
