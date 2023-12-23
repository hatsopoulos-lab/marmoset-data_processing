
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:34:53 2023

@author: daltonm
"""

import cv2
import dill
import glob
from os.path import join as pjoin
import toml
import numpy as np
import matplotlib.pyplot as plt

project_path = '/project/nicho/data/marmosets/kinematics_videos/moth/JLTY/2023_08_03'

def load_detections_and_videos(project_path):
    with open(pjoin(project_path, 'calibration', 'detections.pickle'), 'rb') as f:
        detections = dill.load(f) 

    vidpaths = sorted(glob.glob(pjoin(project_path, 'calibration', '*.avi')))

    anipose_config = toml.load(pjoin(project_path, 'config.toml')) 
    board_size = tuple(anipose_config['calibration']['board_size'])

    return detections, vidpaths, board_size

def extract_images_from_video(vidpath, vid_detections):
    images_to_extract = [info['framenum'][-1] for info in vid_detections]
        
    vidcap = cv2.VideoCapture(vidpath)
        
    success=True
    count=0
    image_list = []
    while success:
        success, image = vidcap.read()
        if count in images_to_extract:
            image_list.append(image)
        
        count+=1
            
    return image_list

def fix_detected_corners_arrangement(image, board_size, corners, filled, correct_arr):

    if correct_arr.lower() == 'r':
        rearr_idx = np.arange(corners.shape[0])[::-1]
    elif correct_arr.lower() == 'v':
        rearr_idx = np.array([], dtype=int)
        for row in range(board_size[1]):
            rearr_idx = np.hstack((np.arange(board_size[0]*row, board_size[0]*(row+1)), 
                                   rearr_idx))
    elif correct_arr.lower() == 'h':
        rearr_idx = np.array([], dtype=int)
        for row in range(board_size[1]):
            rearr_idx = np.hstack((rearr_idx,
                                   np.arange(board_size[0]*(row+1)-1, board_size[0]*row-1, -1)))

    corners = corners[rearr_idx] 
    filled  = filled[rearr_idx] 
    
    # print(corners[:9].squeeze())
    
    image_grid = cv2.drawChessboardCorners(image, board_size, corners, True)
    
    plt.imshow(image_grid)
    plt.pause(.01)
    correct_arr = input('Is this FLIPPED grid correct now? Type [Enter/r/v/h/d] for [Good/Reverse/VertFlip/HorzFlip/Delete]')

    return correct_arr, corners, filled

def manually_correct_calibration_detections(image_list, vid_detect, board_size):
    corrected_video_detections = []
    for image, info in zip(image_list, vid_detect):
        
        corrected_info = info.copy()

        corners = corrected_info['corners']
        filled  = corrected_info['filled']
        
        # print(corners[:9].squeeze())
        
        image_grid = cv2.drawChessboardCorners(image, board_size, corners, True)
        
        plt.imshow(image_grid)
        plt.show()
        plt.pause(.01)
        correct_arr = input('Is the grid correct now? Type [Enter/r/v/h/d] for [Good/Reverse/VertFlip/HorzFlip/Delete]')
        
        while not (len(correct_arr) == 0 or correct_arr.lower() in ['v', 'h', 'r', 'd']):
            correct_arr = input('The option you have entered is not supported. Type [Enter/r/v/h/d] for [Good/Reverse/VertFlip/HorzFlip/Delete]')    
        
        while correct_arr.lower() in ['v', 'h', 'r']:
            correct_arr, corners, filled = fix_detected_corners_arrangement(image, board_size, corners, filled, correct_arr)
        
        if correct_arr.lower() == 'd':
            continue
        
        corrected_info['corners'] = corners
        corrected_info['filled' ] = filled
        
        corrected_video_detections.append(corrected_info)
                
    return corrected_video_detections

if __name__ == '__main__':
    
    detections, vidpaths, board_size = load_detections_and_videos(project_path)
    
    corrected_detections = [[] for i in range(len(detections))]
    for camIdx, (vidpath, vid_detect) in enumerate(zip(vidpaths, detections)):

        print(vidpath)
            
        image_list = extract_images_from_video(vidpath, vid_detect)
        corrected_detections[camIdx] = manually_correct_calibration_detections(image_list, vid_detect, board_size)
        
        
    with open(pjoin(project_path, 'calibration', 'detections.pickle'), 'wb') as f:
        dill.dump(corrected_detections, f, recurse=True)
    