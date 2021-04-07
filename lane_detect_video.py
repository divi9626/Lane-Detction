# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 13:10:12 2021

@author: divyam
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy

########### helper functions #######################
#------------------------------------------------------------------------------
def yellow_mask(image):
    lower = np.uint8([60, 150, 120])
    upper = np.uint8([115, 155, 200])
    yellow_mask = cv.inRange(image, lower, upper)
    return yellow_mask
#------------------------------------------------------------------------------
def mask_white(image):
    lower = np.uint8([180, 180, 180])
    upper = np.uint8([255, 255, 255])
    white_mask = cv.inRange(image, lower, upper)
    return white_mask
#------------------------------------------------------------------------------
def mask_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([190, 190, 190])
    upper = np.uint8([255, 255, 255])
    white_mask = cv.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([60, 150, 120])
    upper = np.uint8([115, 155, 200])
    yellow_mask = cv.inRange(image, lower, upper)
    # combine the mask
    mask = cv.bitwise_or(white_mask, yellow_mask)
    masked = cv.bitwise_and(image, image, mask = mask)
    return masked
#------------------------------------------------------------------------------
def get_turn(actual_lines):
    if actual_lines is None:
        return ['keeping straight']
    if actual_lines[1][0] == 0:
        return ['turning right']
    x1,y1,x2,y2 = actual_lines[0]
    x1_,y1_,x2_,y2_ = actual_lines[1]
    left_slope = abs((y1-y2)/(x1-x2))
    right_slope = abs((y1_-y2_)/(x1_-x2_))
    print(right_slope,left_slope)
    if left_slope > right_slope:
        return ['turning left']
    if left_slope < right_slope:
        return ['turning right']
    else:
        return ['keeping straight']
    pass
#------------------------------------------------------------------------------
def get_coordinates(line_image, params):
    try:
        slope, intercept = params
    except: 
        slope, intercept = 0,0
    if slope == 0 or intercept == 0:
        return np.array([0,0,0,0])
    y1 = line_image.shape[0]
    y2 = int(y1*(1/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    if x1 + x2 > 1000000000 or x1 + x2 < -100000000:
        return np.array([0,0,0,0])
    
    return np.array([x1, y1, x2, y2])
#------------------------------------------------------------------------------
def avg_slope(line_image,lines):
    left_lines = []
    right_lines = []
    if lines is None:                               ## for null output
        return lines
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)           ## extracting points
        params = np.polyfit((x1,x2),(y1,y2),1)
        slope = params[0]
        intercept = params[1]
        print(params,x2)
        if  x2 > 220:
            if slope < 5 and slope > -5:
                continue
            if slope > 40:
                continue
            else:
                right_lines.append((slope,intercept))
        if x2 < 90 and x2 > 10:
            left_lines.append((slope,intercept))    ## averaging lines
    left_avg = np.average(left_lines, axis=0)
    right_avg = np.average(right_lines, axis=0)
    left_line = get_coordinates(line_image, left_avg)
    right_line = get_coordinates(line_image, right_avg)
    print(right_line)
    return np.array([left_line,right_line])
#------------------------------------------------------------------------------
###############################################################################
    
def output(image):
    #--------------------------------------------------------------------------
    K = np.array([[  1.15422732e+03 ,  0.00000000e+00 ,  6.71627794e+02],
     [  0.00000000e+00 ,  1.14818221e+03 ,  3.86046312e+02],
     [  0.00000000e+00 ,  0.00000000e+00 ,  1.00000000e+00]])
    #Distortion coeffecient
    dist = np.array([[ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05, 2.20573263e-02]])
    undist_image = cv.undistort(image,K,dist)
    #--------------------------------------------------------------------------
    ## denoising ##
    smooth_image = cv.GaussianBlur(undist_image,(7,7),0)
    #smooth_image = cv.bilateralFilter(undist_image,5,50,50)
    #smooth_image = cv.medianBlur(undist_image,5)
    #--------------------------------------------------------------------------
    ## cropped image ##
    
    cropped_img = copy.deepcopy(smooth_image)
    cropped_img = cropped_img[430:,:]
    roi = np.zeros_like(image)
    my_roi = roi[120:,280:580]
    #--------------------------------------------------------------------------
    ## homography and warping ##
    
    src_pts = np.array([[540, 65],
                       [780, 65],
                       [1040, 230],
                       [300, 230]], dtype= "float32")
    dest_pts = np.array([[0, 0],
                        [300, 0],
                        [300, 600],
                        [0, 600]], dtype= "float32")
    
    H = cv.findHomography(src_pts,dest_pts)
    M = H[0]
    M_inv = np.linalg.inv(M) 
    warped_image = cv.warpPerspective(cropped_img, M, (my_roi.shape[1], my_roi.shape[0]))
    
    #-------------------------------------------------------------------------- 
    ## Masking ##
    #[-------------------------------------------------------------------------
    #masked_image = mask_rgb_white_yellow(warped_image)
    masked_image = mask_rgb_white_yellow(warped_image)
    #inverse_warp = cv.warpPerspective(warped_image,M_inv,(cropped_img.shape[1],cropped_img.shape[0]))
    #--------------------------------------------------------------------------
    ## edge detection ##
    #--------------------------------------------------------------------------
    _,thresh = cv.threshold(masked_image,125,255, cv.THRESH_BINARY_INV)
    edge_image = cv.Canny(thresh,50,150)
    print(image.shape)
    #--------------------------------------------------------------------------
    ## curve fitting ##
    lines = cv.HoughLinesP(edge_image,2,np.pi/60,60,np.array([]), minLineLength=20, maxLineGap = 50)
    #--------------------------------------------------------------------------    
    ## display lines
    line_image = np.zeros_like(my_roi)
    actual_lines = avg_slope(line_image,lines)
    if actual_lines is not None:
        for line in actual_lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image,(x1,y1),(x2,y2),(0,0,255),2)
    #--------------------------------------------------------------------------
    ## averaging hough lines ##
    if actual_lines is not None:
        if actual_lines[1][0] != 0:
            x1, y1, x2, y2 = actual_lines[0][0],actual_lines[0][1],actual_lines[0][2],actual_lines[0][3]
            x1_, y1_, x2_, y2_ = actual_lines[1][0],actual_lines[1][1],actual_lines[1][2],actual_lines[1][3]
            ind = np.array([[[x1,y1],[x1_,y1_],[x2_,y2_],[x2,y2]]], dtype = np.int32)
    
            line_image = cv.fillConvexPoly(line_image,ind,255)
    #--------------------------------------------------------------------------        
    ## unwarping lines ## 
    warp_with_lines = cv.addWeighted(line_image,0.8,line_image,1,1)
    inverse_warp = cv.warpPerspective(warp_with_lines,M_inv,(cropped_img.shape[1],cropped_img.shape[0]))
    #--------------------------------------------------------------------------
    ## showing lines on unwarped frme ##
    fore = np.zeros_like(frame)
    h = image.shape[0]
    fore[h-cropped_img.shape[0]:,:,:] = inverse_warp
    result = cv.addWeighted(image, 1, fore, 0.8, 1)
    #--------------------------------------------------------------------------
    
    ## Turn Prediction ##
    #--------------------------------------------------------------------------
    if actual_lines is not None:
        if actual_lines[1][0] != 0:
            turn = get_turn(actual_lines)[0]
            
            font = cv.FONT_HERSHEY_SIMPLEX 
            fontScale = 1.0
            # Blue color in BGR 
            color = (0, 0, 255) 
            # Line thickness of 2 px 
            thickness = 2
            # Using cv2.putText() method 
            image = cv.putText(result, turn, (30,30), font,  
                           fontScale, color, thickness, cv.LINE_AA)
    return result, inverse_warp, masked_image, warped_image
###############################################################################
cap = cv.VideoCapture("challenge_video.mp4")
if cap.isOpened() == False:
    print("Error opening the image")
    
count = 0

while cap.isOpened():
    count  += 1
    ret, frame = cap.read()
    if ret == False:
        break
    image, edge, mask, warp = output(frame)
    if count == 80:
        image = frame
    
    cv.imshow('image',image)
    cv.imshow('lanes', edge)
    cv.imshow('mask', mask)
    cv.imshow('warp', warp)
    
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()

###############################################################################


