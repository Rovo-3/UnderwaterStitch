from stitching import Stitcher
import cv2
import glob
import datetime

settings = {"detector": "orb", 
            "confidence_threshold": 0.2, 
            # "blender_type": "feather",
            # "crop": False,
            # "no-crop": "True",
            "wave_correct_kind": "auto"
            }
stitcher = Stitcher(**settings)

image_paths = glob.glob('../../Images/60fps_office/*.png')
images = []

for image in image_paths:
    img = cv2.imread(image)
    print("openting ", image)
    images.append(img)

print("stitching...")
stitched_img = stitcher.stitch(images)

now = datetime.datetime.now()
imgnameprefix = now.strftime("%m-%d-%Y-%H-%M-%S")
imgname = "../../Results/" + imgnameprefix + "_stitched.png"

print("writing ", imgname)
cv2.imwrite(imgname, stitched_img)

cv2.namedWindow('Stitched Img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stitched Img', 1280, 720)
cv2.imshow("Stitched Img", stitched_img)
cv2.waitKey(0)