import cv2
from natsort import natsorted
import glob

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def detectBlur(img, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurlevel = variance_of_laplacian(gray)
    text = "Not Blurry"
    if blurlevel < thresh:
        text = "Blurry"
    return blurlevel, text

imagePaths = natsorted(list(glob.glob("../../Images/officeshelfmasjadonpong/*")))

for imagePath in imagePaths:
    image = cv2.imread(imagePath)    
    fm, text = detectBlur(image, 110)
    
    print(imagePath)
    
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    key = cv2.waitKey(0)