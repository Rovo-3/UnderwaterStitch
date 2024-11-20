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

        if isinstance(score, list) and isinstance(best_score, list):
            # print(1)
            if img_idx not in excluded_images and score[0] > best_score[0]:
                best_match = img_idx
                best_score = score

        elif isinstance(best_score, list):
            # print(2)
            if img_idx not in excluded_images and score > best_score[0]:
                best_match = img_idx
                best_score = score

        elif isinstance(score, list):
            # print(3)
            if img_idx not in excluded_images and score[0] > best_score:
                best_match = img_idx
                best_score = score
        else:
            # print(4)
            # print(f"score {score} :: best_score {best_score}")
            if img_idx not in excluded_images and score > best_score:
                best_match = img_idx
                best_score = score
            # print(f"score {score} :: best_score {best_score}")

    # print(f"return best match : {best_match}")
    return best_match


def generateOrderedImages(matrix):
    current_image = random.choice(range(len(matrix)))

    print(f"Initial center image: {current_image}")

    order = [current_image]

    while len(order) < len(matrix):
        # print(f"({len(order)}/{len(matrix)})")
        # time.sleep(1)
        # print(f"matx {len(matrix)}")
        # print(order)

        left_best = findBestMatch(order[0], order, matrix)
        right_best = findBestMatch(order[-1], order, matrix)

        # print(f"left {left_best} :: right {right_best}")

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
            # continue

    # print("return order")
    return order


def showFinalImage(stitched_image):
    stitched_image = ip.resizeImage(stitched_image, 0.5)
    cv2.namedWindow("stitched_output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("stitched_output", 1280, 720)
    cv2.imshow("stitched_output", stitched_image)
    cv2.waitKey(0)


method = "bf"
ip = ImageProcessor()
dmc = DetectMatchConfidence()
sc = StitchCentral()
ppi = ProcessPairingImage(method=method, dmc=dmc)

path = "./Images/2ndfloor/*"
# path = "./Images/TestSet/set4-rovPool/*"
# path = "./imgTest"
imagePaths = natsorted(list(glob.glob(path)))

arrimgname, arrimage = [], []
arrkeypoints, arrdescriptors = [], []
# newOrderIdx = []
# matxConf = []

heightth = []
widthth = []

sift = cv2.SIFT.create()  # cv.NORM_L2
orb = cv2.ORB.create()  # cv2.NORM_HAMMING
brisk = cv2.BRISK.create()  # cv.NORM_L2
akaze = cv2.AKAZE.create()  # cv.NORM_L2

# datapath = "./2ndfloor.npz"
# data = np.load(datapath)

if __name__ == "__main__":
    print(f"Images Path: {path} || Method: {method}")
    print("loading images..")

    detector = brisk

    start = time.time()
    for image in imagePaths:
        # for image in data.files:
        readImage = cv2.imread(image)
        # readImage = data[image]
        # readImage = ip.rotateImage(readImage)
        readImage = ip.resizeImage(readImage)
        readImage = ip.imagePreProcess(readImage)

        blurlevel = ip.detectBlur(readImage)
        if blurlevel < 100:
            print(blurlevel)
            continue

        imageKeypoint, imageDescriptor = dmc.detectorKeypoint(detector, readImage)

        arrimage.append(readImage)
        arrimgname.append(image)
        arrdescriptors.append(imageDescriptor)
        arrkeypoints.append(imageKeypoint)

        # print(f"Extracting Images: ({len(arrimage)}/{len(imagePaths)})")

    totalimage = len(arrimage)
    totaliterations = totalimage * (totalimage - 1)

    print(f"Time loading iamges: {time.time()-start}")
    print(f"Total Image In Folder: {len(imagePaths)}")
    print(f"Blur Images: {len(imagePaths) - totalimage}")
    print(f"Total Images Now: {totalimage}")

    start = time.time()

    if totaliterations > 5000:
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
    time.sleep(3)

    # print(matxConf)

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
    start = time.time()
    stitched_image = sc.stitchCentral(detector, newImageOrder, allowedPixels=5)
    print(f"Time stitching: {time.time()-start}")

    cv2.imwrite(
        f"./Results/{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}_st_{process}_{method}.png",
        stitched_image,
    )
    print(
        f"./Results/{datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}_st_{process}_{method}.png"
    )

    # print(heightth)
    print()
    # print(widthth)

    print("=== DONE ===")
    print(f"Process : ")
    print(f"Time elapsed: {time.time()-programstart}")

    # showFinalImage(stitched_image)
