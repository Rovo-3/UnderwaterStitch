from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from pathlib import Path
from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching.warper import Warper
from stitching.timelapser import Timelapser
from stitching.cropper import Cropper
from stitching.seam_finder import SeamFinder
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.blender import Blender

# My Code ==================================================================================================================================================================
from natsort import natsorted
import glob
# My Code ==================================================================================================================================================================

def plot_image(img, figsize_in_inches=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def plot_images(imgs, figsize_in_inches=(5, 5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def get_image_paths(img_set):
    return [str(path.relative_to(".")) for path in Path("imgs").rglob(f"{img_set}*")]


# My Code ==================================================================================================================================================================
def wb_opencv(img):
    wb = cv.xphoto.createSimpleWB()
    wb_img = wb.balanceWhite(img)
    return wb_img

def chanelClahe(channel):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    channel_clahe = clahe.apply(channel)
    return channel_clahe

def imagePreProcess(imagetobeprocessed):
    white_balanced_img = wb_opencv(imagetobeprocessed)
    imagesprocessed = imagetobeprocessed
    
    lab_image = cv.cvtColor(imagesprocessed, cv.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv.split(lab_image)

    l_channel_clahe = chanelClahe(l_channel)
    merged_lab = cv.merge((l_channel_clahe, a_channel, b_channel))

    final_img_lab = cv.cvtColor(merged_lab, cv.COLOR_LAB2BGR)
    return final_img_lab

def resizeImage(img, scale):
    h, w, _ = img.shape
    h = h*scale
    w = w*scale
    # print (image.shape)
    # print("Height: ", h, "; Width: ", w)
    img = cv.resize(img, (int(w) , int(h)))
    return img
# My Code ==================================================================================================================================================================

# ==========================================================================================================================================================================
weir_imgs = get_image_paths("weir")
budapest_imgs = get_image_paths("buda")
exposure_error_imgs = get_image_paths("exp")
barcode_imgs = get_image_paths("barc")
barcode_masks = get_image_paths("mask")
# office_imgs = [
#     # str(path.relative_to(".")) for path in Path("../Images/officemanyimages").rglob(f"*")
# ]

# My Code ==================================================================================================================================================================
imagesarr = []
# imagePaths = natsorted(list(glob.glob("../Images/vidframes/*")))
imagePaths = natsorted(list(glob.glob("./dummy/*")))

for imagePath in imagePaths:
    image = cv.imread(imagePath)
    image = resizeImage(image, 1)
    
    processedimage = imagePreProcess(image)
    imagesarr.append(processedimage)
    print(imagePath)
# My Code ==================================================================================================================================================================
# ==========================================================================================================================================================================
images = Images.of(imagesarr)

medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
low_imgs = list(images.resize(Images.Resolution.LOW))
final_imgs = list(images.resize(Images.Resolution.FINAL))
# ==========================================================================================================================================================================
original_size = images.sizes[0]
medium_size = images.get_image_size(medium_imgs[0])
low_size = images.get_image_size(low_imgs[0])
final_size = images.get_image_size(final_imgs[0])

print(
    f"Original Size: {original_size}  -> {'{:,}'.format(np.prod(original_size))} px ~ 1 MP"
)
print(
    f"Medium Size:   {medium_size}  -> {'{:,}'.format(np.prod(medium_size))} px ~ 0.6 MP"
)
print(f"Low Size:      {low_size}   -> {'{:,}'.format(np.prod(low_size))} px ~ 0.1 MP")
print(f"Final Size:    {final_size}  -> {'{:,}'.format(np.prod(final_size))} px ~ 1 MP")
# ==========================================================================================================================================================================
finder = FeatureDetector()
features = [finder.detect_features(img) for img in medium_imgs]
keypoints_center_img = finder.draw_keypoints(medium_imgs[1], features[1])
# ==========================================================================================================================================================================
plot_image(keypoints_center_img, (15, 10))
# ==========================================================================================================================================================================
matcher = FeatureMatcher()
matches = matcher.match_features(features)
# ==========================================================================================================================================================================
matcher.get_confidence_matrix(matches)
# ==========================================================================================================================================================================
all_relevant_matches = matcher.draw_matches_matrix(
    medium_imgs, features, matches, conf_thresh=0.2, inliers=True, matchColor=(0, 255, 0)
)

for idx1, idx2, img in all_relevant_matches:
    print(f"Matches Image {idx1+1} to Image {idx2+1}")
    plot_image(img, (20, 10))
# ==========================================================================================================================================================================
subsetter = Subsetter()
dot_notation = subsetter.get_matches_graph(images.names, matches)
print(dot_notation)
# ==========================================================================================================================================================================
indices = subsetter.get_indices_to_keep(features, matches)

medium_imgs = subsetter.subset_list(medium_imgs, indices)
low_imgs = subsetter.subset_list(low_imgs, indices)
final_imgs = subsetter.subset_list(final_imgs, indices)
features = subsetter.subset_list(features, indices)
matches = subsetter.subset_matches(matches, indices)

images.subset(indices)

print(images.names)
print(matcher.get_confidence_matrix(matches))
# ==========================================================================================================================================================================
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector

camera_estimator = CameraEstimator()
camera_adjuster = CameraAdjuster()
wave_corrector = WaveCorrector()

cameras = camera_estimator.estimate(features, matches)
cameras = camera_adjuster.adjust(features, matches, cameras)
cameras = wave_corrector.correct(cameras)
# ==========================================================================================================================================================================
warper = Warper()
# ==========================================================================================================================================================================
warper.set_scale(cameras)
# ==========================================================================================================================================================================
low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)  # since cameras were obtained on medium imgs

warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)
# ==========================================================================================================================================================================
final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)

warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)
# ==========================================================================================================================================================================
plot_images(warped_low_imgs, (10,10))
plot_images(warped_low_masks, (10,10))
# ==========================================================================================================================================================================
print(final_corners)
print(final_sizes)
# ==========================================================================================================================================================================
timelapser = Timelapser('as_is')
timelapser.initialize(final_corners, final_sizes)

for img, corner in zip(warped_final_imgs, final_corners):
    timelapser.process_frame(img, corner)
    frame = timelapser.get_frame()
    plot_image(frame, (10,10))
# ==========================================================================================================================================================================
cropper = Cropper()
# ==========================================================================================================================================================================
mask = cropper.estimate_panorama_mask(warped_low_imgs, warped_low_masks, low_corners, low_sizes)
plot_image(mask, (5,5))
# ==========================================================================================================================================================================
lir = cropper.estimate_largest_interior_rectangle(mask)
# ==========================================================================================================================================================================
lir = cropper.estimate_largest_interior_rectangle(mask)
print(lir)
# ==========================================================================================================================================================================
plot = lir.draw_on(mask, size=2)
plot_image(plot, (5,5))
# ==========================================================================================================================================================================
low_corners = cropper.get_zero_center_corners(low_corners)
rectangles = cropper.get_rectangles(low_corners, low_sizes)

plot = rectangles[1].draw_on(plot, (0, 255, 0), 2)  # The rectangle of the center img
plot_image(plot, (5,5))
# ==========================================================================================================================================================================
overlap = cropper.get_overlap(rectangles[1], lir)
plot = overlap.draw_on(plot, (255, 0, 0), 2)
plot_image(plot, (5,5))
# ==========================================================================================================================================================================
intersection = cropper.get_intersection(rectangles[1], overlap)
plot = intersection.draw_on(warped_low_masks[1], (255, 0, 0), 2)
plot_image(plot, (2.5,2.5))
# ==========================================================================================================================================================================
cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

cropped_low_masks = list(cropper.crop_images(warped_low_masks))
cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

lir_aspect = images.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL)  # since lir was obtained on low imgs
cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes, lir_aspect)
# ==========================================================================================================================================================================
timelapser = Timelapser('as_is')
timelapser.initialize(final_corners, final_sizes)

for img, corner in zip(cropped_final_imgs, final_corners):
    timelapser.process_frame(img, corner)
    frame = timelapser.get_frame()
    plot_image(frame, (10,10))
# ==========================================================================================================================================================================
seam_finder = SeamFinder()

seam_masks = seam_finder.find(cropped_low_imgs, low_corners, cropped_low_masks)
seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_final_masks)]

seam_masks_plots = [SeamFinder.draw_seam_mask(img, seam_mask) for img, seam_mask in zip(cropped_final_imgs, seam_masks)]
plot_images(seam_masks_plots, (15,10))
# ==========================================================================================================================================================================
compensator = ExposureErrorCompensator()

compensator.feed(low_corners, cropped_low_imgs, cropped_low_masks)

compensated_imgs = [compensator.apply(idx, corner, img, mask) 
                    for idx, (img, mask, corner) 
                    in enumerate(zip(cropped_final_imgs, cropped_final_masks, final_corners))]
# ==========================================================================================================================================================================
blender = Blender()
blender.prepare(final_corners, final_sizes)
for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
    blender.feed(img, mask, corner)
panorama, _ = blender.blend()
# ==========================================================================================================================================================================
plot_image(panorama, (20,20))
# figsize_in_inches=(20,20)

# fig, ax = plt.subplots(figsize=figsize_in_inches)
# ax.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
# cv.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
# cv.waitKey(0)
# # ==========================================================================================================================================================================
# blended_seam_masks = seam_finder.blend_seam_masks(seam_masks, final_corners, final_sizes)
# plot_image(blended_seam_masks, (5,5))
# # ==========================================================================================================================================================================
# plot_image(seam_finder.draw_seam_lines(panorama, blended_seam_masks, linesize=3), (15,10))
# plot_image(seam_finder.draw_seam_polygons(panorama, blended_seam_masks), (15,10))
# # ==========================================================================================================================================================================
