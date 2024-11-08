import cv2
import numpy as np


class StitchCentral:
    def __init__(self):
        print()

    def overlayImages(self, base_img, new_img):
        assert (
            base_img.shape == new_img.shape
        ), "Images must have the same shape for overlaying."

        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

        mask = gray > 0

        base_img[mask] = new_img[mask]

        return base_img

    def canvasSize(self, images, homographies):
        height, width = images[0].shape[:2]

        all_corners = []

        for i, homography in enumerate(homographies):
            corners = np.array(
                [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
            ).reshape(-1, 1, 2)

            transformed_corners = cv2.perspectiveTransform(corners, homography)
            all_corners.append(transformed_corners)

        all_corners = np.concatenate(all_corners, axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

        return x_min, y_min, x_max, y_max

    def stitchCentral(self, detector, images, allowedPixels=4):
        matcher = cv2.NORM_HAMMING if detector == cv2.ORB.create() else cv2.NORM_L2

        central_idx = len(images) // 2

        homographies = [np.eye(3) for _ in range(len(images))]
        print("left")
        for i in range(central_idx - 1, -1, -1):
            kp1, des1 = detector.detectAndCompute(images[i], None)
            kp2, des2 = detector.detectAndCompute(images[i + 1], None)

            bf = cv2.BFMatcher(matcher, crossCheck=True)
            matches = bf.match(des1, des2)

            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )

            homography, _ = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, allowedPixels
            )

            # findTransformed(i, homography)

            homographies[i] = homographies[i + 1] @ homography

        print("right")
        for i in range(central_idx + 1, len(images)):
            kp1, des1 = detector.detectAndCompute(images[i], None)
            kp2, des2 = detector.detectAndCompute(images[i - 1], None)

            bf = cv2.BFMatcher(matcher, crossCheck=True)
            matches = bf.match(des1, des2)

            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )

            homography, _ = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, allowedPixels
            )

            # findTransformed(i, homography)

            homographies[i] = homographies[i - 1] @ homography

        print("canvas")
        x_min, y_min, x_max, y_max = self.canvasSize(images, homographies)
        canvas_width, canvas_height = x_max - x_min, y_max - y_min

        translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        adjusted_homographies = [translation_matrix @ h for h in homographies]

        stitched = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        print("overlay")
        for i in range(len(images)):

            warped_image = cv2.warpPerspective(
                images[i],
                adjusted_homographies[i],
                (canvas_width, canvas_height),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_TRANSPARENT,
            )

            stitched = self.overlayImages(stitched, warped_image)

        return stitched
