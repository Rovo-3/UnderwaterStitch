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
from ProcessPairingImage import ProcessPairingImage


programstart = time.time()

def findBestMatch(current_image, excluded_images, matrix):
    best_match = None
    best_score = -1
    for img_idx, score in enumerate(matrix[current_image]):

        if isinstance(score, list) and isinstance(best_score, list):
            if img_idx not in excluded_images and score[0] > best_score[0]:
                best_match = img_idx
                best_score = score

        elif isinstance(best_score, list):
            if img_idx not in excluded_images and score > best_score[0]:
                best_match = img_idx
                best_score = score

        elif isinstance(score, list):
            if img_idx not in excluded_images and score[0] > best_score:
                best_match = img_idx
                best_score = score
        else:
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

            if left_score >= right_score:
                order.insert(0, left_best)

            else:
                order.append(right_best)

        elif left_best is not None:
            order.insert(0, left_best)

        elif right_best is not None:
            order.append(right_best)

        elif left_best is None or right_best is None:
            print("Loop Error, left or right best is none")
            print(order)
            break

    return order


def showFinalImage(stitched_image):
    stitched_image = ip.resizeImage(stitched_image, 0.5)
    cv2.namedWindow("stitched_output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("stitched_output", 1280, 720)
    cv2.imshow("stitched_output", stitched_image)
    cv2.waitKey(0)


method = "knn"
ordered = False
ip = ImageProcessor()
dmc = DetectMatchConfidence()
sc = StitchCentral()
sc.seamless = True
ppi = ProcessPairingImage(method=method)

path = "./imgTest"
imagePaths = natsorted(list(glob.glob(path)))

arrimgname, arrimage = [], []
arrkeypoints, arrdescriptors = [], []

sift = cv2.SIFT.create()  # cv.NORM_L2
orb = cv2.ORB.create()  # cv2.NORM_HAMMING
brisk = cv2.BRISK.create()  # cv.NORM_L2
akaze = cv2.AKAZE.create()  # cv.NORM_L2

if __name__ == "__main__":
    print(f"Images Path: {path} || Method: {method}")
    print("loading images..")

    detector = brisk

    start = time.time()
    for image in imagePaths:
        readImage = cv2.imread(image)
        readImage = ip.resizeImage(readImage)
        readImage = ip.imagePreProcess(readImage)

        blurlevel = ip.detectBlur(readImage)
        if blurlevel < 200:
            print(blurlevel)
            continue

        imageKeypoint, imageDescriptor = dmc.detectorKeypoint(detector, readImage)

        arrimage.append(readImage)
        arrimgname.append(image)
        arrdescriptors.append(imageDescriptor)
        arrkeypoints.append(imageKeypoint)

    totalimage = len(arrimage)
    totaliterations = totalimage * (totalimage - 1)

    print(f"Time loading iamges: {time.time()-start}")
    print(f"Total Image In Folder: {len(imagePaths)}")
    print(f"Blur Images: {len(imagePaths) - totalimage}")
    print(f"Total Images Now: {totalimage}")

    start = time.time()

    if ordered == False:
        if totaliterations > 1500:
            process = "Parallel"
            print(process)
            matxConf = ppi.imageOrderByConf_Multi(
                totalimage, arrimgname, arrdescriptors, arrkeypoints
            )

        else:
            process = "Single"
            print(process)
            matxConf = ppi.imageOrderByConf_Single(
                totalimage, arrimgname, arrdescriptors, arrkeypoints
            )

        print(f"Time confidence ordered: {time.time()-start}")

        newOrderIdx = generateOrderedImages(matxConf)
        print(newOrderIdx)

        print("copying order")
        newImageOrder = [arrimage[i] for i in newOrderIdx]
    else: 
        newImageOrder = arrimage
        process = "AlreadyOrdered"

    """
    # allowedPixels are the distance of error between two points of matches. 
    # for bigger images, it can use to near 0 numbers
    # for normal images, it can use numbers ranging from 10 ~ 20
    # the general allowedPixels used for every pictures default at 4
    """

    print("central")
    start = time.time()
    stitched_image = sc.stitchCentral(detector, newImageOrder, allowedPixels=5)
    print(f"Time stitching: {time.time()-start}")
    print(f"Feathering: {sc.seamless}")

    cv2.imwrite(
        f"./Results/{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}_st_{process}_{method}_Feather{sc.seamless}.png",
        stitched_image,
    )

    print(
        f"./Results/{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}_st_{process}_{method}_Feather{sc.seamless}.png"
    )

    print()

    print("=== DONE ===")
    print(f"Process : ")
    print(f"Time elapsed: {time.time()-programstart}")

    showFinalImage(stitched_image)
