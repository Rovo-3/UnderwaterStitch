import cv2
import numpy as np

# Read the image
image_path = '../../Images/seaTrial30pics/1.png'
# image_path = '../../SS.png'
image = cv2.imread(image_path)
image = cv2.resize(image, (1280 , 720))
# Images\seaTrial30pics\1.png

# Step 1: Apply Dynamic Threshold White Balance
def dynamic_threshold_white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def wb_opencv(img):
    wb = cv2.xphoto.createSimpleWB()
    wb_img = wb.balanceWhite(img)
    return wb_img

# white_balanced_img = dynamic_threshold_white_balance(image)
white_balanced_img = wb_opencv(image)

# Step 2: Convert the image from BGR to HSV color space
hsv_image = cv2.cvtColor(white_balanced_img, cv2.COLOR_BGR2HSV)
bw_image = cv2.cvtColor(white_balanced_img, cv2.COLOR_BGR2GRAY)
lab_image = cv2.cvtColor(white_balanced_img, cv2.COLOR_BGR2LAB)

# Split the HSV image into H, S, and V channels
h_channel, s_channel, v_channel = cv2.split(hsv_image)
l_channel, a_channel, b_channel = cv2.split(lab_image)

# Step 3: Apply CLAHE to the V channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
v_channel_clahe = clahe.apply(v_channel)
l_channel_clahe = clahe.apply(l_channel)

# Merge the CLAHE enhanced V channel back with H and S channels
hsv_image_clahe = cv2.merge((h_channel, s_channel, v_channel_clahe))
lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

# Step 4: Normalize the HUE channel
h_channel_normalized = cv2.normalize(h_channel, None, alpha=0, beta=179, norm_type=cv2.NORM_MINMAX)

# Merge the normalized H channel back with S and V channels
hsv_image_normalized = cv2.merge((h_channel_normalized, s_channel, v_channel_clahe))

# Convert the HSV image back to BGR color space
final_hsv = cv2.cvtColor(hsv_image_normalized, cv2.COLOR_HSV2BGR)

# final_image2 = clahe.apply(white_balanced_img) + 30
final_bw = clahe.apply(bw_image)
final_lab = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)

# =================================================================


# Convert the image from BGR to YCbCr color space
ycbcr_image = cv2.cvtColor(white_balanced_img, cv2.COLOR_BGR2YCrCb)

# Split the YCbCr image into Y, Cb, and Cr channels
y_channel, cb_channel, cr_channel = cv2.split(ycbcr_image)

y_channel_clahe = clahe.apply(y_channel)

# Merge the CLAHE enhanced Y channel back with Cb and Cr channels
ycbcr_image_clahe = cv2.merge((y_channel_clahe, cb_channel, cr_channel))

# Convert the YCbCr image back to BGR color space
final_ycbcr = cv2.cvtColor(ycbcr_image_clahe, cv2.COLOR_YCrCb2BGR)

# Display the original and enhanced images
cv2.imshow('Original Image', image)
cv2.imshow('CLAHE', lab_image_clahe)
cv2.imshow('Enhanced Image', final_lab)
cv2.waitKey(0)
cv2.destroyAllWindows()