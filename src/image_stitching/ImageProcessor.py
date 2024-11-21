import cv2


class ImageProcessor:
    def __init__(self):
        print()
        
    def rotateImage(self, img):
        rotatedimage = cv2.rotate(img, cv2.ROTATE_180)
        return rotatedimage

    def wb_opencv(self, img):
        wb = cv2.xphoto.createSimpleWB()
        wb_img = wb.balanceWhite(img)
        return wb_img

    def chanelClahe(self, channel):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        channel_clahe = clahe.apply(channel)
        return channel_clahe

    def imagePreProcess(self, imagetobeprocessed):
        imagetobeprocessed = self.wb_opencv(imagetobeprocessed)

        lab_image = cv2.cvtColor(imagetobeprocessed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        l_channel_clahe = self.chanelClahe(l_channel)

        merged_lab = cv2.merge((l_channel_clahe, a_channel, b_channel))
        final_img_lab = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        return final_img_lab

    def resizeImage(self, img, scale=1):
        h, w, _ = img.shape
        h = int(h * scale)
        w = int(w * scale)
        resized_img = cv2.resize(img, (w, h))
        return resized_img

    def detectBlur(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurlevel = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blurlevel
