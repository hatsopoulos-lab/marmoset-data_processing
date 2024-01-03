#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:08:57 2023

@author: daltonm
"""

from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget
import ndx_pose
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
from importlib import sys, reload
from pathlib import Path
sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import read_prb_hatlab, plot_prb

processed=False
marmscode = 'JLTY'
nwb_acquisition_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/JL20231126_1325_foraging_day4/JL20231126_1325_foraging_day4002_acquisition.nwb'
nwb_processed_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/MG20230416_1505_mothsAndFree/MG20230416_1505_mothsAndFree-002_processed_OLD_NEUROCONV.nwb'

kinematics_video_path = Path('/project/nicho/data/marmosets/kinematics_videos/')

date_pattern = re.compile('[0-9]{8}_')

def validate_acquisition_nwb(nwb):
    print(nwb)
    
    es_key = [key for key in nwb.acquisition.keys() if 'Electrical' in key][0]

    # create timestamps for raw neural data from starting_time, rate, and data shape
    start = nwb.acquisition[es_key].starting_time
    step = 1/nwb.acquisition[es_key].rate
    stop = start + step*nwb.acquisition[es_key].data.shape[0]
    print(f'sample rate = {nwb_acq.acquisition[es_key].rate} kHz')
    print(f'start time in neural data is {start} sec')
    print(f'stop time in neural data is {stop} sec')
    
    elec_df = nwb.electrodes.to_dataframe()
    analog_idx = [idx for idx, name in elec_df['electrode_label'].items() if 'ainp' in name]
    array_idx = [idx for idx, name in elec_df['electrode_label'].items() if 'elec' in name]
    
    print('\n-------------------------------------------------------------------\n')
    
    print(nwb.acquisition['screenshots of neural data acquisition'])
    for img_name, img in nwb.acquisition['screenshots of neural data acquisition'].images.items():
        print(f'plotting {img_name}')
        plt.imshow(img)
        plt.show()
        
    print('\n-------------------------------------------------------------------\n')
    
    print(nwb.acquisition['neural signal dropout plots'])
    for img_name, img in nwb.acquisition['neural signal dropout plots'].images.items():
        print(f'plotting {img_name}')
        plt.imshow(img, interpolation='nearest')
        plt.show()

    print('\n-------------------------------------------------------------------\n')

    print(f'Channel info:\n keys = {elec_df.columns}')
    print(f'shape = {elec_df.shape}, with analog inputs in channels {analog_idx} and\n array data in channels {array_idx}')

    print('\n-------------------------------------------------------------------\n')
    
    date = re.findall(date_pattern, nwb.identifier)[0][:8]
    date = f'{date[:4]}_{date[4:6]}_{date[6:]}'
    timestamps_keys = [key for key in nwb.processing.keys() if 'timestamps' in key]
    for tKey in timestamps_keys:
        experiment_video_path = list(kinematics_video_path.glob(f'*{tKey.split("timestamps_")[-1]}'))[0] / marmscode / date / 'avi_videos'
        print(f'\nThere are {len(nwb.processing[tKey].data_interfaces)} events stored in processing-->{tKey}')
        iKey = [key for key in nwb.intervals.keys() if key.split('video_events_')[-1] == tKey.split('video_event_timestamps_')[-1]][0] 
        for idx, (key, values) in enumerate(nwb.processing[tKey].data_interfaces.items()):
            sess  = re.findall(re.compile('_s_[0-9]'), key)[0][-1]        
            event = re.findall(re.compile('_s_[0-9]_e_[0-9]{3}'), key)[0][-3:]
            video_paths = sorted(list(experiment_video_path.glob(f'*s{sess}_e{event}*.avi')))
            frameCounts = []                  
            for vid in video_paths:
                cap = cv2.VideoCapture(str(vid))
                frameCounts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            videos_match_nwb_timestamps = [True if ct == values.timestamps.size else False for ct in frameCounts]
            if all(videos_match_nwb_timestamps):
                print(f'  {key}: frame count = {values.timestamps.size:>6}, startTime = {np.round(nwb.intervals[iKey].start_time[idx], 2):>7.2f}, stopTime = {np.round(nwb.intervals[iKey].stop_time[idx], 2):>7.2f} --- All {len(frameCounts)} video frame counts match timestamps.')
            else:
                print(f'  {key}: frame count = {values.timestamps.size:>6}, startTime = {np.round(nwb.intervals[iKey].start_time[idx], 2):>7.2f}, stopTime = {np.round(nwb.intervals[iKey].stop_time[idx], 2):>7.2f} --- Video frame counts = {frameCounts}')

        dropKey = [key for key in nwb.processing.keys() if key.split('frames_')[-1] == tKey.split('timestamps_')[1]][0]
        if len(nwb.processing[dropKey].data_interfaces) == 0:
            print(f'\nNo dropped frames in any of {tKey}')
        else:
            print(f'\nFound dropped frames in {tKey}')
            for drop_mask_key, drop_mask in nwb.processing[dropKey].data_interfaces.items():
                print(f'     {drop_mask_key}:  n(dropped_frames) = {drop_mask.data.size - np.sum(drop_mask.data[:])}')
            
            
        calibration_video_path = list(kinematics_video_path.glob(f'*{tKey.split("timestamps_")[-1]}'))[0] / marmscode / date / 'calibration'
        calib_video_paths = sorted(list(calibration_video_path.glob('*.avi')))
        frameCounts = []                  
        for vid in calib_video_paths:
            cap = cv2.VideoCapture(str(vid))
            frameCounts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))        
        print(f'\nExperiment = {tKey.split("timestamps_")[-1]}. Calibration video frame counts = {frameCounts}')
        print('\n-------------------------------------------------------------------\n')

    iKey = 'neural_dropout'
    print('\nNeural Dropout intervals')
    print(nwb.intervals[iKey].to_dataframe())
    
    return elec_df
    
with NWBHDF5IO(nwb_acquisition_file, mode='r') as io_acq:
    nwb_acq = io_acq.read()

    elec_df = validate_acquisition_nwb(nwb_acq)

    if processed:
        with NWBHDF5IO(nwb_processed_file, mode='r') as io_prc:
            nwb_prc = io_prc.read()