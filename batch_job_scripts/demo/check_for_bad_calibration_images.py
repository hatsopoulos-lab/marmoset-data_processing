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

project_path = '/project/nicho/data/marmosets/kinematics_videos/moths/HMMG/2023_04_16'

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

def fix_detected_corners_arrangement(image, board_size, corners, filled):

    rearr_idx = np.arange(corners.shape[0])[::-1]

    corners = corners[rearr_idx] 
    filled  = filled[rearr_idx] 
    
    image_grid = cv2.drawChessboardCorners(image, board_size, corners, True)
    
    plt.imshow(image_grid)
    plt.pause(.01)
    correct_arr = input('Is this FLIPPED grid correct now? Type [Enter/f/d] for [Good/Flip/Delete]')

    return correct_arr, corners, filled

def manually_correct_calibration_detections(image_list, vid_detect, board_size):
    corrected_video_detections = []
    for image, info in zip(image_list, vid_detect):
        
        corrected_info = info.copy()

        corners = corrected_info['corners']
        filled  = corrected_info['filled']
        
        image_grid = cv2.drawChessboardCorners(image, board_size, corners, True)
        
        plt.imshow(image_grid)
        plt.show()
        plt.pause(.01)
        correct_arr = input('Is the grid correct now? (Blue top left, Red bottom right) Type [Enter/f/d] for [Good/Flip/Delete]')
        
        while correct_arr.lower() == 'f':
            correct_arr, corners, filled = fix_detected_corners_arrangement(image, board_size, corners, filled)
        
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
    