#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:12:43 2024

@author: daltonm
"""

# TODO
'''
    Need to implement checks for instances where there is no good reference camera.
    Could do this by comparing all the questionable endpoints to event_idx=0,
    and instead of comparing to aground truth reference length you would find the event  
    index for each camera such that the "time_to_event_start" value matches across all
    cameras. The event index will be differenct across the cameras.
'''

import numpy as np
import pandas as pd
from pathlib import Path
import os
import dill
import re
from importlib import sys

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import save_dict_to_hdf5, load_dict_from_hdf5

session=1
date='2023_11_23'
exp = 'foraging'
marms = 'JLTY'
intermediate_data_path = Path(f'/project/nicho/data/marmosets/kinematics_videos/{exp}/{marms}/{date}/manual_corrections_records/session{session}_episode_correction_record.h5')
jpg_dir = Path(f'/scratch/midway3/daltonm/kinematics_jpgs/{exp}/{marms}/{date}/session{session}/')
fps = 200
all_cams_list = [1, 2, 3, 4, 5]

event_pattern      = re.compile('event_\d{3,5}_')
start_time_pattern = re.compile('\d{4}-\d{2}.jpg')
timestamp_pattern  = re.compile('currentTime_\d{9,20}')
framenum_pattern   = re.compile('frame_\d{7}')
 
def measure_event_length_for_each_potential_endpoint(jpg_dir, ref_cam, ref_cam_events_to_jump, start_event_idx, cams_to_fix, potential_next_event_idxs, fps): 
    
    cam_jpg_dir = jpg_dir / f'jpg_cam{ref_cam}'
    event_glob      = f'*event_{str(start_event_idx+1).zfill(3)}_*'
    next_event_glob = f'*event_{str(start_event_idx+1+ref_cam_events_to_jump).zfill(3)}_*'
    start_event_frames = sorted(list(cam_jpg_dir.glob(event_glob)))
    next_event_frames = sorted(list(cam_jpg_dir.glob(next_event_glob)))
    first_timestamp = int(re.findall(timestamp_pattern, start_event_frames[0].stem)[0].split('currentTime_')[-1])
    next_first_timestamp  = int(re.findall(timestamp_pattern,  next_event_frames[0].stem)[0].split('currentTime_')[-1])
    ref_event_length = np.round((next_first_timestamp - first_timestamp) * 1e-9, 3)
    
    potential_next_event_starts = dict()
    for cIdx, (cam_num, events_list) in enumerate(zip(cams_to_fix, potential_next_event_idxs)):
        cam_jpg_dir = jpg_dir / f'jpg_cam{cam_num}'
        event_glob = f'*event_{str(start_event_idx+1).zfill(3)}_*'
        start_event_frames = sorted(list(cam_jpg_dir.glob(event_glob)))
        start_event_first_timestamp = int(re.findall(timestamp_pattern, start_event_frames[ 0].stem)[0].split('currentTime_')[-1])
        
        time_diffs_to_event_start = []
        time_diffs_to_event_end = []
        event_idx_list = []
        event_start_times = []
        ref_length_list = []
        for eIdx in events_list:
            event_num = eIdx + 1
            event_glob = f'*event_{str(event_num).zfill(3)}_*'
            print(cam_jpg_dir / event_glob)
            
            frames = sorted(list(cam_jpg_dir.glob(event_glob)))            
            if len(frames) > 0:
                start_times = [re.findall(start_time_pattern, frame.name)[0].split('.jpg')[0] for frame in frames]
                if len(start_times) > 0:
                    if len(np.unique(start_times)) > 1:
                        print('multiple start times')
                    else:
                        event_start_times.append(start_times[0])
                
                first_timestamp = int(re.findall(timestamp_pattern, frames[ 0].stem)[0].split('currentTime_')[-1])
                last_timestamp  = int(re.findall(timestamp_pattern, frames[-1].stem)[0].split('currentTime_')[-1])
                
                time_diffs_to_event_start.append(np.round((first_timestamp - start_event_first_timestamp) * 1e-9, 3))
                time_diffs_to_event_end.append  (np.round((last_timestamp  - start_event_first_timestamp) * 1e-9, 3))
                
                ref_length_list.append(ref_event_length)
                event_idx_list.append(eIdx)
        
        potential_next_event_starts[f'cam_{cam_num}'] = pd.DataFrame(data = zip(event_idx_list, 
                                                                                event_start_times, 
                                                                                ref_length_list,
                                                                                time_diffs_to_event_start, 
                                                                                time_diffs_to_event_end),
                                                                     columns = ['eIdx', 
                                                                                'start_time', 
                                                                                'ref_time',
                                                                                'time_to_event_start', 
                                                                                'time_to_event_end'])

    return potential_next_event_starts


if __name__ == '__main__':

    # Load existing data or gather metadata from all files, then save. 
    inter_data = load_dict_from_hdf5(intermediate_data_path)
    
    potential_next_event_starts = measure_event_length_for_each_potential_endpoint(jpg_dir, 
                                                                                   ref_cam=1, 
                                                                                   start_event_idx=19,
                                                                                   ref_cam_events_to_jump=2,
                                                                                   cams_to_fix=[3, 4, 5], 
                                                                                   potential_next_event_idxs= [range(443, 800), range(460, 800), range(158, 400)], 
                                                                                   fps=fps)
                                 
