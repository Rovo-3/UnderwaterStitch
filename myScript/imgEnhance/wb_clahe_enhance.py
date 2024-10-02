import cv2
import datetime

image_path = '../../Images/coralblurry/4.png'
# image_path = '../cameraframes/Frame13.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (854 , 480))

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", image)
cv2.waitKey(0)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.imshow("color", image)
cv2.waitKey(0)


now = datetime.datetime.now()
imgnameprefix = now.strftime("%m-%d-%Y-%H-%M-%S")
imgname = imgnameprefix + "_WB_CLAHE.png"
# outpath = "../../WB_CLAHE_IMG/" + imgname
outpath = "../../Results/" + imgname

def wb_opencv(img):
    wb = cv2.xphoto.createSimpleWB()
    wb_img = wb.balanceWhite(img)
    return wb_img

def chanelClahe(channel):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    channel_clahe = clahe.apply(channel)
    return channel_clahe
    
white_balanced_img = wb_opencv(image)

lab_image = cv2.cvtColor(white_balanced_img, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(lab_image)

l_channel_clahe = chanelClahe(l_channel)
merged_lab = cv2.merge((l_channel_clahe, a_channel, b_channel))

final_img_lab = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

cv2.imshow("Original Image", image)
cv2.imshow("White Balanced Image", white_balanced_img)
cv2.imshow("CLAHE Merged", l_channel_clahe)
cv2.imshow("Final Image", final_img_lab)

cv2.imwrite(outpath, final_img_lab)

cv2.waitKey(0)