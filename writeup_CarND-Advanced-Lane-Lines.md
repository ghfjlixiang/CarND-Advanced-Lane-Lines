---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./writeup_images/binary_combo_example.jpg "Binary Example"
[image4]: ./writeup_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./writeup_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./writeup_images/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines **26** through **53** of the file called `utils.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Firstly,I pickled the camera calibration and distortion coefficients,mtx and dst ,Then I used them in cv2.undistort to remove distortion in the test images. The code for this step is contained in code cell **6** of the IPython notebook located in "./CarND-Advanced-Lane-Lines.ipynb".See example below of a distortion corrected image
<img src="./writeup_images/undist_test1.png" width="600">

#### 2. Use color transforms, gradients or other methods to create a thresholded binary image. 

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 124 through 173 in `utils.py`).  Firstly,I used the x gradient and the gradient direction thresholds at the same time. But I found that gradient thresholding using sobel operator in x-direction was the most effective. Then for RGB color space, I combined the R and G channel thresholds,so that Both white and yellow lanes can be detected well. Next for HLS color space thresholding, I used the s-channel to identify yellow and white lines better under different lightening conditions,the S channel is still doing a fairly robust job of picking up the lines under very different color and contrast conditions, while the other selections look messy. You could tweak the thresholds and get closer in the L channels, but the S channel is preferable because it is more robust to changing conditions. Further I combined the gradient and the R&G channel and the L&S channel thresholds to obtain the thresholded binary image. Finally,I apply the region of interest mask to the combined thresholded binary image,So that the interference from invalid areas can be excluded.Here's an example of my output for this step.  
![alt text][image3]

#### 3. Perform a perspective transform.
After thresholding,I perform the perspective transform to change the image to bird's eye view which let’s us view a lane from above; this will be useful for calculating the lane curvature later on.
The code for my perspective transform includes a function called `warper()`, which appears in lines 175 through 205 in the file `utils.py` (./utils.py) .  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identify lane-line pixels and fit their positions with a polynomial
After applying calibration, thresholding, and a perspective transform to a road image, Now I have a binary image where the lane lines stand out clearly. Firstly,I use the two most prominent peaks in a Histogramas as a starting point for where to search for the lines and to split the histogram for the two lines.Next,I set up windows and  appropriate window hyperparameters,then iterating through nwindows to track curvature and extract left and right line pixel positions in function `find_lane_pixels`(coded in lines 207 through 295 in the file `utils.py`).Finally, I fit a 2nd order polynomial curve to all the relevant pixels that founded in sliding windows in `fit_poly` function (coded in lines 297 through 320 in the file `utils.py`).However, using the sliding window to finding the lines and starting fresh on every frame may seem inefficient, as the lines don't necessarily move a lot from frame to frame.So, once we know where the lines are in one frame of video, you can use the previous polynomial to skip the sliding window and do a highly targeted search for them in the next frame,I implemented this Search from Prior method in function `search_around_poly`(Coded in the code cell **17** of the IPython notebook).
Here's an example of my output for this step:
![alt text][image5]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
In the case of the second order polynomial,the radius of curvature be calculated from the first and second derivatives.I evaluate the radius of curvature for left and right lanes at the bottom of image respectively.
Also it's important to note that you have to define conversions in x and y from pixels space to meters before you start the calculation.

Since we can assume that the camera is mounted at the center of the car ,such that the lane center is the midpoint at the bottom of the image between the two lines detected. The offset of the lane center from the center of the image (converted from pixels to meters) is the distance from the center of the lane. 

I did this in lines 346 through 370 in my code in `utils.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I Warp the lane area back to original image space using inverse perspective matrix (Minv).I implemented this step in lines 372through 390 in my code in `utils.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

After setting up all the functions needed for the Pipeline, I firstly tested a pipeline that combined all the functions on two consecutive frame images that extracted from project video (Coded in the code cell ** 15 ** and ** 18 ** of the IPython notebook). Based on the results of image processing pipeline, and using the tips and tricks (Tracking、Sanity Check、 Search from Prior、Reset、Smoothing etc.) that provided in class for building a robust pipeline,I created a video process pipeline (Coded in the code cell **24** of the IPython notebook) for real-world lane lines identification situaion. Here's a few steps that in my pipeline:
* If it is processing the first picture or the lane line recognition of last frame is failure then the pipeline will start searching from scratch using a histogram and sliding window

* Otherwise the previous left and right polynomial fit are used to search in a margin around the previous line position and identify lane lines efficiently (function implemented in code cell **17**),ignoring starting fresh on every frame and iterating through the windows( fit_skip = True).If Search from Prior is failure,the pipeline will re-use Sliding Window method instead.

* If the left and right lane lines fit successfully,before moving on, it should check that the detection makes sense. To confirm that your detected lane lines are real, you might consider checking if they have similar curvature?If they are separated by approximately the right distance horizontally?If they are roughly parallel?
According to the Sanity Check results, Determining if a second lane line identification and Sanity Check is required?

* If the Sanity Check results is OK, the pipeline will append new fits to the list of recent measurements and then take an average over n past measurements to obtain the lane position to draw onto the image.So here I've defined a Line() class to keep track of all the interesting parameters you measure from frame to frame (class defined in code cell **22**) and a Smoothing function `get_averaged_line` (defined in code cell **23**) to smooth over the last n frames of video to obtain a cleaner result. 

* If the left and right lane lines Fitting failure or the Sanity Check results is NG, then the last good values of left and right fits were used.

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems/issues that I have faced were:

1. At the beginning, after the search around method failed to fit, my pipeline would not adopt the sliding window method to search again immediately, and the sanity check was not strict. So if the road environment changes dramatically, it is easy to cause the lane line detection results to deviate from the actual situation, and even fail to recover in the end.
2. In the project video,The left yellow lane line is easily confused with the boundary line at the junction of the lane and guardrail,since the histogram have more than two peak. To solve this I introduced threshold judgment of R&G channel in threshold binarization of image.

As you can see the scheme above relies heavily on tuning the parameters for thresholding/warping and those can be camera/track specific. So here are a few  other situations where the pipeline might fail:
1. It is easy to be influenced by this method to detect the lane line in the road sections with dense traffic flow and people flow, and the reliability is not high
2. Presence of snow/debris on the road that makes it difficult to clearly see lane lines
3. Roads that are wavy or S shaped where a second order polynomial may not be able to properly fit lane lines

To make the code more robust we should try to incorporate the ability to test a wide range of parameters (for thresholding and warping) so that the code can generalize to a wider range of track and weather conditions. To do this we need a very powerful sanity check function that can identify lane lines that are impractical and discard them.