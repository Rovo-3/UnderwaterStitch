import cv2
import numpy as np
from datetime import datetime

# Function to apply dynamic white balance
def dynamic_white_balance(image):
    wb = cv2.xphoto.createSimpleWB()
    corrected_image = wb.balanceWhite(image)
    return corrected_image

# Function for CLAHE contrast enhancement
def apply_CLAHE(image, clip_limit=2.0, grid_size=(4, 4)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    output = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR) 
    return output

def optical_flow_warp(prev_frame, next_frame):
    # Resize the next frame to match the previous frame size (if needed)
    next_frame = cv2.resize(next_frame, (prev_frame.shape[1], prev_frame.shape[0]))

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Get the height and width of the previous frame
    h, w = prev_gray.shape

    # Create a grid of (x, y) coordinates
    y_indices, x_indices = np.indices((h, w))

    # Create the flow map by adding the flow to the original grid
    flow_map = np.empty((h, w, 2), dtype=np.float32)
    flow_map[..., 0] = x_indices + flow[..., 0]  # x coordinates
    flow_map[..., 1] = y_indices + flow[..., 1]  # y coordinates

    # Warp the next frame to align with the previous frame
    warped_frame = cv2.remap(next_frame, flow_map[..., 0], flow_map[..., 1], cv2.INTER_LINEAR)

    return warped_frame



# Function to stitch the warped images
def stitching_image(fileName, currentImage):
    image = cv2.imread(fileName)
    try:
        if image is None:
            cv2.imwrite(fileName, currentImage)
        else:
            stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
            stitcher.setFeaturesFinder(cv2.ORB_create())
            (status, stitched) = stitcher.stitch([image, currentImage])
            if status == 0:
                print("Done Stitching")
                cv2.imwrite(fileName, stitched)
                cv2.imshow("stitched", stitched)
            else:
                print("Failed Stitching")
                global frame_count 
                frame_count = -1
    except Exception as e:
        print(f"Stitching failed: {e}")

def get_center_roi(image, roi_width_ratio=0.8, roi_height_ratio=0.8):
    (h, w) = image.shape[:2]
    roi_w = int(w * roi_width_ratio)
    roi_h = int(h * roi_height_ratio)
    x_start = (w - roi_w) // 2
    y_start = (h - roi_h) // 2
    roi = image[y_start:y_start + roi_h, x_start:x_start + roi_w]
    return roi

# Capture video
cam = cv2.VideoCapture("./videos/VID1.mp4")

frame_count = 0
frame_skip = 1
prev_frame = None  # To store previous frame for optical flow

now = datetime.now()
date = now.strftime('%y%m%d_%H%M%S')
filename = "./output/webcam/stitchedimage" + str(date) + ".jpg"

while True:
    ret, cap = cam.read()
    if not ret:
        break
    
    image = get_center_roi(cap, 0.4, 0.4)

    # Optional white balance and CLAHE
    # image = dynamic_white_balance(image)
    # image = apply_CLAHE(image)
    
    # Apply optical flow if there's a previous frame
    if prev_frame is not None:
        image = optical_flow_warp(prev_frame, image)
        cv2.imshow("Warped Image", image)

    if frame_count % frame_skip == 0:
        stitching_image(fileName=filename, currentImage=image)
    
    # Update previous frame
    prev_frame = image.copy()
    frame_count += 1

    # Display video
    cv2.imshow("Video", cap)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
