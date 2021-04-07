# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 13:10:12 2021

@author: divyam
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy


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
    if x1 + x2 > 1000000000 or x1 + x2 < -100000000:           ## thresholding values
        return np.array([0,0,0,0])
    
    return np.array([x1, y1, x2, y2])

def avg_slope(line_image,lines):
    left_lines = []
    right_lines = []
    if lines is None:
        return lines
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1,x2),(y1,y2),1)
        slope = params[0]
        intercept = params[1]
        #print(params,x2)
        if slope < 0:
            right_lines.append((slope,intercept))
        if slope > 0:
            left_lines.append((slope,intercept))
    left_avg = np.average(left_lines, axis=0)
    right_avg = np.average(right_lines, axis=0)
    left_line = get_coordinates(line_image, left_avg)
    right_line = get_coordinates(line_image, right_avg)
    return np.array([left_line,right_line])

def lane_detect(image):
    ## unidistoring the image ##
    #Camera Matrix
    K = np.array([[  1.15422732e+03 ,  0.00000000e+00 ,  6.71627794e+02],
     [  0.00000000e+00 ,  1.14818221e+03 ,  3.86046312e+02],
     [  0.00000000e+00 ,  0.00000000e+00 ,  1.00000000e+00]])
    #Distortion coeffecient
    dist = np.array([[ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05, 2.20573263e-02]])
    undist_image = cv.undistort(image,K,dist)
    
    ## denoising ##
    smooth_image = cv.GaussianBlur(undist_image,(7,7),0)
    #smooth_image = cv.bilateralFilter(undist_image,5,50,50)
    #smooth_image = cv.medianBlur(undist_image,5)
    
    ## FINDING ROI ##
    cropped_img = copy.deepcopy(smooth_image)
    cropped_img = cropped_img[250:490,:]
    
    
    ## edge detection ##
    gray_smooth = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
    _,thresh = cv.threshold(gray_smooth,230,255, cv.THRESH_BINARY_INV)
    edge_image = cv.Canny(thresh,40,200)
    
    ## Hough lines ##
    lines = cv.HoughLinesP(edge_image,6,np.pi/180,150,np.array([]), minLineLength=50, maxLineGap = 50)
    
    ## display lines
    line_image = np.zeros_like(cropped_img)
    actual_lines = avg_slope(line_image,lines)
    if actual_lines is not None:    
        if actual_lines[1][0] != 0:
            x1, y1, x2, y2 = actual_lines[0][0],actual_lines[0][1],actual_lines[0][2],actual_lines[0][3]
            x1_, y1_, x2_, y2_ = actual_lines[1][0],actual_lines[1][1],actual_lines[1][2],actual_lines[1][3]
            ind = np.array([[[x1,y1],[x1_,y1_],[x2_,y2_],[x2,y2]]], dtype = np.int32)
        
            line_image = cv.fillConvexPoly(line_image,ind,255)
            
    if actual_lines is not None:
        for line in actual_lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
            
    ## merge display with original image ##
    #merge_image = cv.addWeighted(cropped_img,1,line_image,1,1)
    
    fore = np.zeros_like(image)
    h = image.shape[0]
    fore[h-cropped_img.shape[0]:,:,:] = line_image
    result = cv.addWeighted(image, 1, fore, 0.8, 1)
    
    return result, line_image, edge_image, thresh

cap = cv.VideoCapture("project.avi")
if cap.isOpened() == False:
    print("Error opening the image")
    
count = 0

while cap.isOpened():
    count  += 1
    ret, frame = cap.read()
    if ret == False:
        break
    
    image, lines, canny, thresh = lane_detect(frame)
#    if count == 50:
#        image = frame
    
    cv.imshow('image',image)
    cv.imshow('lines',lines)
    cv.imshow('edges', canny)
    cv.imshow('thresh', thresh)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()

#cv.imshow('img' , merge_image)
#cv.waitKey(0)
#cv.destroyAllWindows()
