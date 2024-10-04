import cv2
import numpy as np
from imutils import paths
import time

def knn_matcher(des1, des2):
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    ratio_thresh = 0.75
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches

def stitch_images_with_knn(images):
    orb = cv2.ORB_create()
    keypoints_descriptors = []

    for image in images:
        keypoints, descriptors = orb.detectAndCompute(image, None)
        keypoints_descriptors.append((keypoints, descriptors))

    stitched_image = images[0]

    for i in range(1, len(images)):
        kp1, des1 = keypoints_descriptors[i-1]
        kp2, des2 = keypoints_descriptors[i]

        good_matches = knn_matcher(des1, des2)

        if len(good_matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Check if H is valid
            if H is not None and H.shape == (3, 3):
                # Get the dimensions of the images
                height, width = stitched_image.shape[:2]
                new_width = width + images[i].shape[1]
                new_height = max(height, images[i].shape[0])

                # Create a canvas for the stitched image
                canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)

                # Place the stitched image on the canvas
                canvas[0:height, 0:width] = stitched_image

                # Warp the next image to align with the current stitched image
                warped_image = cv2.warpPerspective(images[i], H, (new_width, new_height))

                # Combine the current stitched image and the new warped image
                canvas = np.maximum(canvas, warped_image)
                stitched_image = canvas
            else:
                print(f"[WARNING] Homography matrix is invalid between image {i-1} and image {i}.")
        else:
            print(f"[WARNING] Not enough matches between image {i-1} and image {i}. Skipping this pair.")
            continue

    return stitched_image



def main():
    # Load the input images
    image_paths = sorted(list(paths.list_images("./images/set11")))
    images = [cv2.imread(imagePath) for imagePath in image_paths]

    print("[INFO] Stitching images with KNN-based feature matching...")
    stitched_image = stitch_images_with_knn(images)

    if stitched_image is not None:
        date = time.time()
        output_filename = "./output/stitchedimage_" + str(date) + ".jpg"
        cv2.imwrite("./output.jpg", stitched_image)
        print(f"[INFO] Image stitching complete. Output saved to {output_filename}")
        cv2.imshow("Stitched Image", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[INFO] Image stitching failed.")

if __name__ == "__main__":
    main()
