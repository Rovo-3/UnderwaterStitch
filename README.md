# UnderwaterStitch
This Repo is for Image Stitching for Underwater Development Project in Rovostech. The project has 2 main development path: Image Stitching and ROV Guidance

Clone the code from this directory for usage.
``` terminal
git clone https://github.com/Rovo-3/UnderwaterStitch
```

## Directory Tree:

Image Stitching, and ROV Guidance and Control program is in the src folder.
```
├───feasibility_study
│   ├───Bagas
│   │   └───myScript
│   │       ├───imgEnhance
│   │       └───Trials
│   └───Jason
│       └───image_stitching
└───src
    ├───image_stitching
    └───lawn_mowing_movement
        └───simulation_and_playground
```
## Image Stitching

[1]: https://docs.opencv.org/3.4/d9/d7a/classcv_1_1xphoto_1_1WhiteBalancer.html
[2]: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
[3]: https://ieeexplore.ieee.org/document/8346440
[4]: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
[5]: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
[6]: https://www.mdpi.com/2076-3417/13/22/12251

Image Stitching code could be found in `src/image_stitching` folder. This Image Stitching program consist of 5 main scripts: 
1. [main.py](./src/image_stitching/main.py) : Importing images and the main part of programs.
2. [ImageProessor.py](./src/image_stitching/ImageProcessor.py) : The pre-processing of images using [White Balance][1] and [CLAHE][2].
3. [ProcessPairingImage.py](./src/image_stitching/ProcessPairingImage.py) : Automatically orders images by features if imported images are not ordered manually. 
4. [DetectMatchConfidence.py](./src/image_stitching/DetectMatchConfidence.py) : Extracts features with [SIFT/ORB/BRISK/AKAZE][3], [KNN/BF][4] matching, [Homography][5] transformation and confidence match between images. 
5. [StitchCentral.py](./src/image_stitching/StitchCentral.py) : Stitch images by using a central image, and [Blending][6] technique. 



Navigate to [image_stitching](./src/image_stitching/), and run the code : 
```
py main.py
```
Customization 
```
main.py

Line
├─89      method      --> "bf" for more precision or "knn" for faster time
├─90      ordered     --> "True" if images are manually ordered or "False" to find image order automatically
├─94      sc.seamless --> "True" if needed feathering or "False" if no feathering
└─98      path        --> "path to images"
```

## Stitch Result

![Corals](./asset/11-21-2024-10-54-45_st_AlreadyOrdered_bf_FeatherTrue.png)
![Bike](./asset/11-21-2024-14-14-56_st_AlreadyOrdered_bf_FeatherTrue.png)

## ROV Guidance and Control
ROV Guidance could be found in the src/lawn_mowing_movement

Overall, the lawn mowing movement consist of 5 main code:
1. Control PID Code [(control.py)](./src/lawn_mowing_movement/control.py)

2. Waypoint Maker for Lawn Mowing [(waypoint_generator.py)](./src/lawn_mowing_movement/waypoint_generator.py)

3. Guidance Code (LOS, PP and Stanley) [(guidance.py)](./src/lawn_mowing_movement/guidance.py)

4. Data Acquisition [(get_data_real.py)](./src/lawn_mowing_movement/get_data_real.py)

5. Main Program to Compile Control, WP, and Guidance [(main.py)](./src/lawn_mowing_movement/main.py)

Settings and datas are stored in JSON file
- parameter.json

    Storing parameter for PID, setpoint and LM waypoint settings. Those parameter can be dynamicly changed during the tuning, and before commanding the LM movement.

    LM_param
    - "heading": desired heading in degree for LM
    - "length": length of each LM
    - "gap" : the gap between LM
    - "iterations" : desired iterations for LM

- sensor_data.json

    Store data from Data Acquisition so that Main Program can know the location and status of ROV


To run the code, navigate to [lawn_mowing_movement directory](./src/lawn_mowing_movement/), run the start.bat file

```console
.\start.bat
```
It will automatically run the main program and the data acquisition program.

Example of LM movement:

![LM movement result](./asset/LM_movement.jpg)

## Future Development
### ROV Guidance: 
1. Optimize the transformation of the coordinates.
2. Create self coordinates based on DVL velocity if option 1 is not visible.
3. Add ability to control altitude.
4. Add feature to enables assist in attitude control. (Currently, joystick must be detached)

### Image Stitching
1. Optimize image projection: APAP, Cropping
2. Faster ordered image processing (GPU)
