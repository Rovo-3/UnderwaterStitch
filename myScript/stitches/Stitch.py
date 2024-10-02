import cv2
import numpy as np

def stitch():
    img1 = cv2.imread("./pic1.jpg")
    img2 = cv2.imread("./pic2.jpg")
    
    grayimg1 = cv2.cvtColor(img1,  cv2.COLOR_BGR2GRAY)
    grayimg2 = cv2.cvtColor(img2,  cv2.COLOR_BGR2GRAY)
    
    # sift = cv2.SIFT_create()
    sift = cv2.SIFT_create()
    
    keypoints1, descriptors1 = sift.detectAndCompute(grayimg1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(grayimg2, None)
    
    peng_match = cv2.BFMatcher()
    banyak_match = peng_match.match(descriptors1,descriptors2)
    
    banyak_match = sorted(banyak_match, key=lambda x: x.distance)
    
    top_matches = banyak_match[:50]
    
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)
    
    matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    result = cv2.warpPerspective(img1, matrix, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result = img2[0:img2.shape[0], 0:img2.shape[1]] 

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', 300, 700)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print()
    
def stitch2():
    # Load the images
    # image1 = cv2.imread("./newspaper1.jpg")
    # image2 = cv2.imread("./newspaper2.jpg")
    
    # image1 = cv2.imread("./5.jpg")
    # image2 = cv2.imread("./6.jpg")
    
    # image1 = cv2.imread("./mount1.jpg")
    # image2 = cv2.imread("./moount2.jpg")

    # image1 = cv2.imread("./first.jpg")
    # image2 = cv2.imread("./second.jpg")
    
    image1 = cv2.imread("./Images/rovdive/Screenshot (49).png")
    image2 = cv2.imread("./Images/rovdive/Screenshot (50).png")
    
    
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create a SIFT object
    sift = cv2.SIFT.create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Select top matches
    top_matches = matches[:50]

    # Extract keypoints from top matches
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

    # Estimate transformation matrix
    matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 10.0)

    # Warp the images
    result = cv2.warpPerspective(image1, matrix, (image1.shape[1] + image2.shape[1], image1.shape[0]))
    result[0:image2.shape[0], 0:image2.shape[1]] = image2

    keypointimage1 = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypointimage2 = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Display the result
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Result', 1280, 720)
    cv2.imshow('Result', result)
    
    cv2.namedWindow('key1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('key1', 1280, 720)
    cv2.imshow('key1', keypointimage1)
    
    cv2.namedWindow('key2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('key2', 1280, 720)
    cv2.imshow('key2', keypointimage2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stitch2()