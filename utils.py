import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_compare(img_src,img_dst,title1='Original Image',title2='Pipeline Result',fontsize=20,figsize=(20,10),save=False,filename=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    f.tight_layout()

    if len(img_src.shape) > 2:
        ax1.imshow(img_src)
    else:
        ax1.imshow(img_src,cmap='gray')        
    ax1.set_title(title1, fontsize=fontsize)

    if len(img_dst.shape) > 2:
        ax2.imshow(img_dst)
    else:
        ax2.imshow(img_dst,cmap='gray')
    ax2.set_title(title2, fontsize=fontsize)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if save == True and filename != None:
        plt.savefig(filename)

def getDistort():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    # Camera calibration, given object points, image points, and the shape of the grayscale image 
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
    return mtx,dist

def getPerspectiveTransform(img_size):
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

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return src,dst,M,Minv

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def threshod(img,color_thresh=150,sx_thresh=(20, 100),dir_thresh=(np.pi/6, np.pi/2),l_thresh=(120, 255),s_thresh=(100, 255),visual=False):
    # Threshold gray
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    ksize = 3
    gradx = abs_sobel_thresh(gray, 'x', sobel_kernel=ksize, thresh=sx_thresh)
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=dir_thresh)
    
    gray_inds = ((gradx == 1) & (dir_binary == 1))
    gray_binary = np.zeros_like(dir_binary)
    gray_binary[gray_inds] = 1
    
    # R & G thresholds so that yellow lanes are detected well.
    R = img[:,:,0]
    G = img[:,:,1]
    rgb_inds = (R > color_thresh) & (G > color_thresh)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # L Channel Threshold x gradient
    l_inds = (l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])
    l_binary = np.zeros_like(l_channel)
    l_binary[l_inds] = 1
    
    # Threshold Saturation channel
    s_inds = (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])
    s_binary = np.zeros_like(s_channel)
    s_binary[s_inds] = 1
                       
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors    
    color_binary = np.dstack((gray_binary,l_binary,s_binary))*255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(gray)
    combined_binary[(rgb_inds & l_inds) & (s_inds | gray_inds)] = 1
    
    # apply the region of interest mask
    mask = np.zeros_like(combined_binary)
    region_of_interest_vertices = np.array([[0,height-1], [width/2, int(0.5*height)], [width-1, height-1]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], 1)
    result = cv2.bitwise_and(combined_binary, mask)*255
    
    if visual == True:
        show_compare(img,color_binary,title1='Undistort Image',title2='Stacked thresholds')
        show_compare(combined_binary,result,title1='Combined thresholds',title2='Masked thresholds')
    return result

def warper(img,dist,mtx,src,dst,M,visual=False):

    undist  = cv2.undistort(img, mtx, dist, None,mtx)
    bin_img = threshod(undist ,visual=False)
    img_size = (bin_img.shape[1], bin_img.shape[0])

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(bin_img, M, img_size,flags=cv2.INTER_LINEAR)

    if visual == True:
        undist_img = np.copy(undist)
        pts = src.reshape((-1, 1, 2))
        cv2.polylines(undist_img, np.int_([pts]), True, (255, 0, 0),3)
        # pts = src.reshape((-1, 2))
        # point_size = 10
        # point_color = (255, 0, 0)
        # thickness = -1
        # for point in pts:
        #     cv2.circle(undist_img, tuple(point), point_size, point_color, thickness)

        color_warped = cv2.warpPerspective(undist,M,img_size,flags=cv2.INTER_LINEAR)
        pts = dst.reshape((-1, 1, 2))
        cv2.polylines(color_warped, np.int_([pts]), True, (255, 0, 0),3)
        # pts = dst.reshape((-1, 2))        
        # for point in pts:
        #     cv2.circle(color_warped, tuple(point), point_size, point_color, thickness)

        show_compare(undist_img,color_warped,title1='Color Undistort Image',title2='Color Warped Image')
        # show_compare(undist_img,color_warped,title1='Undistort Image with src. points drawn',title2=' Warped Result with dest. points drawn',save=True,filename="./writeup_images/warped_straight_lines.jpg")
        show_compare(bin_img,warped,title1='Binary Unistort Image',title2='Binary Warped Image')
    return undist,bin_img,warped

def find_lane_pixels(binary_warped,visual=False):
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    if visual==True:
        fig = plt.figure()
        plt.plot(histogram)
        plt.show()
#         plt.close()

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        if visual == True:
            print('current pos:',leftx_current,rightx_current)    
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 3) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    # If no pixels were found return None
    if(lefty.size == 0 or leftx.size == 0 or righty.size == 0 or rightx.size == 0):
        left_fitx = None
        right_fitx = None        
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])

        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = None
            right_fitx = None
        
    return left_fitx, right_fitx, ploty, left_fit, right_fit



def fit_polynomial(binary_warped,visual=False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped,visual)
    
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    if visual == True:
        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        
        # Plots the left and right polynomials on the lane lines
        plt.figure()
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.imshow(out_img)
        # plt.savefig("./writeup_images/color_fit_lines.jpg")
        show_compare(binary_warped,out_img,title1='Binary Warped Image',title2='Fit Polynomial Image')

    return left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, lefty, rightx, righty, out_img

def measure_curvature_real(img_shape,leftx,rightx,ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)    
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    vehicle_pos = (leftx[-1] + rightx[-1] - img_shape[1])*xm_per_pix/2
    return left_curverad, right_curverad, vehicle_pos

def map_lane(undist,Minv,leftx,rightx,ploty,visual=False):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    if visual==True:
        show_compare(undist,result,title1="Undistort Image",title2="Inverse Warped Image")
    return result