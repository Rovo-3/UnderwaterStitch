import cv2
import numpy as np
 
# Reading the image from the present directory
image_path = '../../Images/GreenClams/2greenResearchClam.jpg'
# image_path = '../../SS.png'
image = cv2.imread(image_path)
# Resizing the image for compatibility
image = cv2.resize(image, (500, 600))
 
# The initial processing of the image
# image = cv2.medianBlur(image, 3)
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h_channel, s_channel, v_channel = cv2.split(image_hsv)
 
# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
# final_img = clahe.apply(image_bw) + 30
v_channel_clahe = clahe.apply(v_channel) + 30
h_channel_normalize = cv2.normalize(h_channel, None, alpha=0, beta=179, norm_type=cv2.NORM_MINMAX)

merge_channel2img = cv2.merge((h_channel_normalize,s_channel,v_channel_clahe))
final_img = cv2.cvtColor(merge_channel2img, cv2.COLOR_HSV2BGR)

# Ordinary thresholding the same image
_, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)
 
# Showing the two images
cv2.imshow("ordinary threshold", ordinary_img)
cv2.imshow("CLAHE image", final_img)
cv2.imshow("Image", image)
cv2.waitKey(0)

# Images\seatrial30pics222\13.png
# Images\GreenClams\2greenResearchClam.jpg