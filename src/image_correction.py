import cv2
import numpy as np

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def color_correction(image):
    # Step 1: Input Image
    original_image = image.copy()

    # Step 2: White Balance Adjustment
    lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab_image)
    
    avg_a = np.mean(A)
    avg_b = np.mean(B)
    
    target_a = 128
    target_b = 128
    
    scale_a = target_a / avg_a if avg_a != 0 else 1
    scale_b = target_b / avg_b if avg_b != 0 else 1
    
    A = np.clip(A * scale_a, 0, 255).astype(np.uint8)
    B = np.clip(B * scale_b, 0, 255).astype(np.uint8)
    
    adjusted_lab_image = cv2.merge([L, A, B])
    color_corrected_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_Lab2BGR)

    # Step 3: Histogram Equalization
    gray_image = cv2.cvtColor(color_corrected_image, cv2.COLOR_BGR2GRAY)
    equalized_gray_image = histogram_equalization(gray_image)
    
    # Apply histogram equalization to L channel
    L_channel = cv2.split(adjusted_lab_image)[0]
    equalized_L = histogram_equalization(L_channel)
    adjusted_lab_image = cv2.merge([equalized_L, A, B])
    
    color_corrected_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_Lab2BGR)

    # Step 4: Color Saturation Adjustment
    hsv_image = cv2.cvtColor(color_corrected_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_image)
    
    saturation_factor = 1.3  # Increase saturation by 30%
    S = np.clip(S * saturation_factor, 0, 255).astype(np.uint8)
    
    adjusted_hsv_image = cv2.merge([H, S, V])
    color_corrected_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)

    # Step 5: Output Image
    return color_corrected_image

# Example usage
image = cv2.imread('./images/set13/A.png')
corrected_image = color_correction(image)

# Display the results
cv2.imwrite('OriginalImage.jpg', image)
cv2.imwrite('ColorCorrectedImage.jpg', corrected_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
