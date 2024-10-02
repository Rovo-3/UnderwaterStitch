import cv2
import glob
import datetime

image_paths = glob.glob('./Images/60fps_office/*.png')
images = []

print("test")

for image in image_paths:
    print(image)
    img = cv2.imread(image)
    images.append(img)

print(images)

imageStitcher = cv2.Stitcher_create()
error, stitched_img = imageStitcher.stitch(images)

if not error:
    now = datetime.datetime.now()
    imgnameprefix = now.strftime("%m-%d-%Y-%H-%M-%S")
    imgname = "./Results/" + imgnameprefix + "_stitched.png"
    print(imgname)
    # cv2.imwrite(str(imgname), stitched_img)
    cv2.imwrite(imgname, stitched_img)
    
    cv2.namedWindow('Stitched Img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stitched Img', 1280, 720)
    cv2.imshow("Stitched Img", stitched_img)
    cv2.waitKey(0)
    
else:
    print("Images could not be stitched!")
    print("Likely not enough keypoints being detected!")