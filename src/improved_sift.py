import cv2
import numpy as np
from skimage import exposure
cv2.WINDOW_NORMAL
# Function for dynamic threshold white balance
def dynamic_white_balance(image):
    wb = cv2.xphoto.createSimpleWB()
    corrected_image = wb.balanceWhite(image)
    cv2.imwrite("ImprovedSIFT_correctedWB.jpg", corrected_image)
    showimg("correctedWB", corrected_image)
    return corrected_image

# Function for CLAHE contrast enhancement
def apply_CLAHE(image, clip_limit=2.0, grid_size=(4, 4)):
    # convert to lab colorspace and spliting it
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # increase the contrast of image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)
    # merge it
    lab_clahe = cv2.merge((l_clahe, a, b))
    output = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR) 
    clahe_img = cv2.imwrite("ImprovedSIFT_clahe.jpg",output)
    showimg("output", output)
    return output

def showimg(name, img):
    img = cv2.resize(img, (0,0), fx=0.5,fy=0.5)
    cv2.imshow(name, img)
    
# Preprocessing step
def preprocess_image(image):
    balanced_image = dynamic_white_balance(image)
    cv2.imwrite("whitebalanced2.jpg",balanced_image)
    enhanced_contrast_image = apply_CLAHE(balanced_image)
    
    return enhanced_contrast_image

# SIFT feature detection
def detect_and_compute_sift(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# KNN feature matching
def knn_feature_matching(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

# RANSAC Homography estimation
def get_homography(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

# Warp and stitch images using homography
def warp_and_stitch(image1, image2, H):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Corners of the images
    corners_image1 = np.float32([[0, 0], [0, height1 - 1], [width1 - 1, height1 - 1], [width1 - 1, 0]]).reshape(-1, 1, 2)
    corners_image2 = np.float32([[0, 0], [0, height2 - 1], [width2 - 1, height2 - 1], [width2 - 1, 0]]).reshape(-1, 1, 2)
    
    transformed_corners_image2 = cv2.perspectiveTransform(corners_image2, H)

    # Stitch the images
    final_corners = np.concatenate((corners_image1, transformed_corners_image2), axis=0)
    [x_min, y_min] = np.int32(final_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(final_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Warp image 2 to fit image 1
    result = cv2.warpPerspective(image2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    y1_end = min(result.shape[0], translation_dist[1] + height1)
    x1_end = min(result.shape[1], translation_dist[0] + width1)

    # Also ensure image1 fits into the space allocated
    y1_image_end = min(height1, result.shape[0] - translation_dist[1])
    x1_image_end = min(width1, result.shape[1] - translation_dist[0])

    # Copy image1 into the corresponding location in the result image
    result[translation_dist[1]:translation_dist[1] + y1_image_end, translation_dist[0]:translation_dist[0] + x1_image_end] = image1[:y1_image_end, :x1_image_end]
    return result

# Main stitching function
def stitch_images(image1, image2):
    # Preprocess the images
    image1_preprocessed = preprocess_image(image1)
    image2_preprocessed = preprocess_image(image2)

    # SIFT feature detection and description
    kp1, des1 = detect_and_compute_sift(image1_preprocessed)
    kp2, des2 = detect_and_compute_sift(image2_preprocessed)

    # KNN feature matching
    matches = knn_feature_matching(des1, des2)

    # RANSAC homography estimation
    H = get_homography(kp1, kp2, matches)

    # Warp and stitch the images
    stitched_image = warp_and_stitch(image1_preprocessed, image2_preprocessed, H)

    return stitched_image

# Example usage
if __name__ == '__main__':
    # Load underwater images
    image1 = cv2.imread('./images/set13/10.png')
    # image2 = cv2.imread('./images/set13/11.png')
    # image color correction and contrast enhancement
    corrected_img = dynamic_white_balance(image1)
    clahceimg = apply_CLAHE(corrected_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Stitch the images
    # result = stitch_images(image1, image2)
    # resultresized = cv2.resize(result, (300,300))
    # # Save or display the result
    # cv2.imwrite('stitched_output.jpg', result)
    # cv2.imshow('Stitched Image', result)
    # cv2.imshow('Stitched Image Resized', resultresized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
