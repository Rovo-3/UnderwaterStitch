import cv2
import numpy as np
import glob
import datetime
import random
import time
from natsort import natsorted


from DetectMatchConfidence import DetectMatchConfidence
from ImageProcessor import ImageProcessor
from StitchCentral import StitchCentral

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

    return order


method = "bf"
ip = ImageProcessor()
dmc = DetectMatchConfidence()
sc = StitchCentral()

# path = "../Images/TestSet/set6-rovpool2/*"
# path = "../Images/TestSet/set5-rovpool/*"
# path = "../myScript/Trials/st2/*"
path = "./imgTest"
imagePaths = natsorted(list(glob.glob(path)))
print(f"Images Path: {path} || Method: {method}")

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


if __name__ == "__main__":
    print("loading images..")
    for image in imagePaths:
        readImage = cv2.imread(image)
        readImage = ip.rotateImage(readImage)
        readImage = ip.resizeImage(readImage)
        readImage = ip.imagePreProcess(readImage)

        blurlevel = ip.detectBlur(readImage)
        if blurlevel < 0:
            print(blurlevel)
            continue

        imageKeypoint, imageDescriptor = dmc.detectorKeypoint(brisk, readImage)

        arrimage.append(readImage)
        arrimgname.append(image)
        arrdescriptors.append(imageDescriptor)
        arrkeypoints.append(imageKeypoint)

        # print(f"Extracting Images: ({len(arrimage)}/{len(imagePaths)})")

    print(f"Total Image In Folder: {len(imagePaths)}")
    print(f"Blur Images: {len(imagePaths) - len(arrimage)}")
    print(f"Total Images Now: {len(arrimage)}")

    for i in range(len(arrimage)):
        arrconfidence = []
        for j in range(len(arrimage)):

            """
            # nmatches a.k.a number of matches, the higher the value will make the stitching and pairing more accurate
            # also note that the bigger the number of nmatches the longer the time will be to be calculated.
            # reccomended is 2000 matches for most images.
            """

            if arrimgname[i] == arrimgname[j]:
                confidence = 0
                arrconfidence.append(int(confidence))
                continue

            # KNN
            if method == "knn":
                try: 
                    matches = dmc.BFMatchKNN(arrdescriptors[i], j, arrdescriptors)
                    confidence = dmc.knnConfidenceMatch(
                        matches, arrkeypoints[i], j, arrkeypoints
                    )
                except:
                    confidence = 0

            # Normal
            elif method == "bf":
                matches = dmc.BFMatch(
                    arrdescriptors[i], j, arrdescriptors, nmatches=500
                )
                confidence = dmc.findConfidenceMatch(
                    matches, arrkeypoints[i], j, arrkeypoints
                )

            arrconfidence.append(int(confidence * 1000))

        print(f"({i+1}/{len(arrimage)})")

        conf = max(arrconfidence)
        confIndex = arrconfidence.index(conf)
        matxConf.append(arrconfidence)

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
    stitched_image = sc.stitchCentral(brisk, newImageOrder, allowedPixels=5)

    cv2.imwrite(
        f"./Results/{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}_st_{method}.png",
        stitched_image,
    )
    print(f"./Results/{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}_st_{method}.png")

    # print(heightth)
    print()
    # print(widthth)

    print("=== DONE ===")
    end = time.time()
    print(f"Time elapsed: {end-start}")

    stitched_image = ip.resizeImage(stitched_image, 0.5)
    cv2.namedWindow("stitched_output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("stitched_output", 1280, 720)
    cv2.imshow("stitched_output", stitched_image)
    cv2.waitKey(0)
