# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 18:30:43 2021

@author: Dalton
"""

import cv2 as cv
import numpy as np
import pandas as pd
import pickle
import os
import glob

operSystem = 'windows' # can be windows or linux

processedData_dir     = '/marmosets/processed_datasets/analog_signal_and_video_frame_information'

kin_archive_path = '/marmosets/kinematics_videos'

marmoset_kinematics_code = 'TYJL' #check in kinematics_videos directory

data_check_file = '20210218_to_20210404_data_to_check.pkl'

class path:
    if operSystem == 'windows':
        base = r'Z:'
        analog_processed_dir = os.path.join(base, processedData_dir)
        
    elif operSystem == 'linux':
        base = '/media/CRI'
        analog_processed_dir = os.path.join(base, processedData_dir)
    
    data_check_file = data_check_file
    
    kinPath = os.path.join(base, kin_archive_path)
        
    del base
    
    
class params:
    nCams = 2
    
#%%
def computeOpticalFlow(videoPath):
    cap = cv.VideoCapture(videoPath)
      
    # ret = a boolean return value from
    # getting the frame, first_frame = the
    # first frame in the entire video sequence
    ret, first_frame = cap.read()
      
    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally 
    # expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
      
    # Creates an image filled with zero
    # intensities with the same dimensions 
    # as the frame
    mask = np.zeros_like(first_frame)
      
    # Sets image saturation to maximum
    mask[..., 1] = 255
    
    store_mag   = []
    store_angle = []
    
    fNum = 0
    while(cap.isOpened()):
          
        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        ret, frame = cap.read()
        fNum += 1
        
        if fNum > 27472 and fNum < 32907:
        
            # # Opens a new window and displays the input
            # # frame
            # cv.imshow("input", frame)
              
            # Converts each frame to grayscale - we previously 
            # only converted the first frame to grayscale
            try:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            except:
                break
            
            print(fNum)
            
            # Calculates dense optical flow by Farneback method
            flow = cv.calcOpticalFlowFarneback(prev_gray, gray, 
                                               None,
                                               0.5, 3, 15, 3, 5, 1.2, 0)
              
            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
              
            # Sets image hue according to the optical flow 
            # direction
            mask[..., 0] = angle * 180 / np.pi / 2
              
            # Sets image value according to the optical flow
            # magnitude (normalized)
            mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
              
            # # Converts HSV to RGB (BGR) color representation
            # rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
              
            # # Opens a new window and displays the output frame
            # cv.imshow("dense optical flow", rgb)
              
            # Updates previous frame
            prev_gray = gray
              
            # Frames are read by intervals of 1 millisecond. The
            # programs breaks out of the while loop when the
            # user presses the 'q' key
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
            magnitude = magnitude.flatten()
            angle = angle.flatten()
            
            top_mag_idx  = np.argsort(magnitude)[-200:]
            top_mags = magnitude[top_mag_idx]
            top_angles = angle[top_mag_idx]
            
            store_mag.append(top_mags.mean())
            store_angle.append(top_angles.mean())
        elif fNum > 32907:
            break
        else: 
            print(fNum)

    # The following frees up resources and
    # closes all windows
    cap.release()
    cv.destroyAllWindows()
    
    return np.array((store_mag, store_angle)).T
    

#%%  load data_check_file and identify which videos to run optical flow

with open(os.path.join(path.analog_processed_dir, path.data_check_file), 'rb') as fp:
    data_to_check = pickle.load(fp)
    
check_idx = [idx for idx, ct in enumerate(data_to_check.counts) if ct[0] != ct[1]]

stored_flows = []
for idx in [check_idx[33]]:
    exp = str(data_to_check.experiment[idx])
    date = str(data_to_check.date[idx])
    sess = str(data_to_check.session[idx])
    event = str(data_to_check.event[idx]+1)
    
    most_frames = int(max(data_to_check.counts[idx][:2]))
    
    flows = pd.DataFrame(np.zeros((most_frames, 4)), columns = ['cam1_mag', 'cam2_mag', 'cam1_angle', 'cam2_angle'])
    flows.iloc[:] = np.nan
    
    vidPath = glob.glob(os.path.join(path.kinPath, 
                                     exp, 
                                     marmoset_kinematics_code, 
                                     date[:4]+'_'+date[4:6]+'_'+date[6:], 
                                     '*_s' + sess + '*_e' + event.zfill(3) + '*.avi'))
    if len(vidPath) == 0:
        vidPath = glob.glob(os.path.join(path.kinPath, 
                                         exp, 
                                         marmoset_kinematics_code, 
                                         date[:4]+'_'+date[4:6]+'_'+date[6:], 
                                         '*_session' + sess + '*_event' + event.zfill(3) + '*.avi'))
    
    
    for cam, vid in enumerate(vidPath):
        flow_out = computeOpticalFlow(vid)
        flows.iloc[:flow_out.shape[0], cam] = flow_out[:, 0]
        flows.iloc[:flow_out.shape[0], params.nCams + cam] = flow_out[:, 1]
