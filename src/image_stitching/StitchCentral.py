import cv2
import numpy as np


class StitchCentral:
    def __init__(self):

        # added by Jason
        self.first = True
        self.offset = 80
        self.seamless = True

    def overlayImages(self, base_img, new_img, masks=None):
        assert (
            base_img.shape == new_img.shape
        ), "Images must have the same shape for overlaying."
        
        base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

        mask = gray > 0
        base_img_ori = base_img.copy()
        base_img[mask] = new_img[mask]
        combined_img = base_img
        
        if self.seamless:
            assert (masks is not None), "Mask must be profided"
            edge_mask, (inner_mask, kernel) = masks

            edge_mask_gray = cv2.cvtColor(edge_mask, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(base_gray, 1, 255, cv2.THRESH_BINARY)

            edge_mask_dilated = cv2.dilate(
                edge_mask_gray, None, iterations=int(self.offset * 2)
            )
            overlap_mask = cv2.bitwise_and(edge_mask_dilated, binary)
            
            featheringTransition = self._feathering(combined_img, base_img_ori, new_img, inner_mask, kernel, overlap_mask)
            
            return featheringTransition

        return combined_img

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
            masks = self.getEdgeMask(warped_image)
            stitched = self.overlayImages(stitched, warped_image, masks)

        return stitched

    def getEdgeMask(self, image):
        # Create empty mask
        mask = np.zeros_like(image, dtype=np.uint8)
        inner_mask_offset = mask.copy()
        edge_mask = mask.copy()

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # tresholding to black and white
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)

        cv2.drawContours(edge_mask, [largest_contour], -1, (255, 255, 255), thickness=1)
        cv2.drawContours(
            inner_mask_offset,
            [largest_contour],
            -1,
            (255, 255, 255),
            thickness=cv2.FILLED,
        )

        inner_mask_offset = cv2.erode(
            inner_mask_offset, None, iterations=int(self.offset)
        )

        size = int(self.offset * 2.8)
        kernel_size = (size + 1, size + 1) if size % 2 == 0 else (size, size)
        # blurred_inner_mask = cv2.GaussianBlur(inner_mask_offset,kernel_size,0)
        return (edge_mask, (inner_mask_offset, kernel_size))

    def gaussianBlurTransition(self, base_img, edge_mask):
        blurred_img = cv2.GaussianBlur(base_img, (99, 99), 0)
        output = np.where(edge_mask == np.array([255, 255, 255]), blurred_img, base_img)
        return output

    def _feathering(self, combined_img, base_img, overlay_img, mask, kernel, overlap_mask = None):
        blurred_mask  = cv2.GaussianBlur(mask,kernel,0)
        feathered_mask_normalized = blurred_mask / 255.0  # Normalize to range [0, 1]

        feathered_image = (
            base_img * (1 - feathered_mask_normalized)
            + overlay_img * feathered_mask_normalized
        ).astype(np.uint8)

        if overlap_mask is not None:
            overlap_mask = cv2.merge([overlap_mask] * 3)
            masked = cv2.bitwise_and(overlap_mask, feathered_image)

            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            mask = gray > 0
            combined_img[mask] = masked[mask]

            feathered_image = combined_img

        return feathered_image
