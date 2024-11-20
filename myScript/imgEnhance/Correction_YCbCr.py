import cv2
import numpy as np

# Read the image
# image_path = '../../Images/GreenClams/2greenResearchClam.jpg'
image_path = '../../SS.png'
image = cv2.imread(image_path)

# Convert the image from BGR to YCbCr color space
ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Split the YCbCr image into Y, Cb, and Cr channels
y_channel, cb_channel, cr_channel = cv2.split(ycbcr_image)

# Apply CLAHE to the Y channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
y_channel_clahe = clahe.apply(y_channel)

# Merge the CLAHE enhanced Y channel back with Cb and Cr channels
ycbcr_image_clahe = cv2.merge((y_channel_clahe, cb_channel, cr_channel))

# Convert the YCbCr image back to BGR color space
final_image = cv2.cvtColor(ycbcr_image_clahe, cv2.COLOR_YCrCb2BGR)

# Display the original and enhanced images
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
