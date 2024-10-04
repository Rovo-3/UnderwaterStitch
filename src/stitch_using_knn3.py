import cv2
import numpy as np

# Load images
img1 = cv2.imread('./images/booth/5.jpg')
img2 = cv2.imread('./images/booth/6.jpg')

# Convert images to grayscale (for SIFT)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Create a FLANN based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(f"Good Matches: {len(good_matches)}")

# Draw matches
match_img = cv2.resize(cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS), (300,300))

cv2.imshow('Good Matches', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


if len(good_matches) > 10:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is not None:
        # Print the homography matrix
        print("Homography Matrix:\n", M)

        # Get dimensions of the images
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        
        # Create an output image with appropriate size
        stitched_width = width1 + width2
        stitched_height = max(height1, height2)
        stitched_img = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

        # Warp image1
        warped_img1 = cv2.warpPerspective(img1, M, (stitched_width, stitched_height))
        warped_img1=cv2.resize(warped_img1,(300,300))
        cv2.imshow('Warped Image 1', warped_img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Place warped image1 into stitched image
        stitched_img[0:warped_img1.shape[0], 0:warped_img1.shape[1]] = warped_img1

        # Place image2 into stitched image, adjusting the region to ensure placement is correct
        stitched_img[0:height2, width1:width1 + width2] = img2

        # Save and show the result
        cv2.imwrite("./output/stitch_image_knn3.jpg", stitched_img)
        cv2.imshow('Stitched Image', stitched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Homography matrix could not be computed.")
else:
    print("Not enough good matches are found.")
