import glob
import cv2
import datetime
from natsort import natsorted


def wbOpenCV(img):
    wb = cv2.xphoto.createSimpleWB()
    wb_img = wb.balanceWhite(img)
    return wb_img


def chanelClahe(channel):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    channel_clahe = clahe.apply(channel)
    return channel_clahe


def imagePreProcess(img):
    white_balanced_img = wbOpenCV(img)
    img = white_balanced_img

    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    l_channel_clahe = chanelClahe(l_channel)
    merged_lab = cv2.merge((l_channel_clahe, a_channel, b_channel))

    final_img_lab = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    return final_img_lab


# path = "../../Images/slrcoralround/*"
path = "./1.jpg"
imagePaths = natsorted(list(glob.glob(path)))

for image in imagePaths:
    now = datetime.datetime.now()
    imgnameprefix = now.strftime("%m-%d-%Y-%H-%M-%S")
    imgname = imgnameprefix + "_WB_CLAHE.png"
    outpath = "../../Results/" + imgname

    readImage = cv2.imread(image)
    white_balanced_img = wbOpenCV(readImage)
    final_img_lab = imagePreProcess(readImage)

    cv2.imshow("Original Image", readImage)
    cv2.imshow("White Balanced Image", white_balanced_img)
    cv2.imshow("Final Image", final_img_lab)

    cv2.imwrite(outpath, final_img_lab)

    cv2.waitKey(1000)
