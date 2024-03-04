#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:12:43 2024

@author: daltonm
"""

'''
    Meant to run in conda environment located at /beagle3/nicho/environments/nwb_and_neuroconv, which uses Python 3.11.5
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

first_run = False

session=1
date='2023_08_09'
exp = 'moth'
marms = 'JLTY'
intermediate_data_path = Path(f'/project/nicho/data/marmosets/kinematics_videos/{exp}/{marms}/{date}/manual_corrections_records/session{session}_episode_correction_record.h5')
jpg_dir = Path(f'/scratch/midway3/daltonm/kinematics_jpgs/{exp}/{marms}/{date}/session{session}/')
fps = 200
all_cams_list = [1, 2, 3, 4, 5]


event_pattern      = re.compile('event_\d{3,5}_')
start_time_pattern = re.compile('\d{4}-\d{2}.jpg')
timestamp_pattern  = re.compile('currentTime_\d{9,20}')
framenum_pattern   = re.compile('frame_\d{7}')

os.makedirs(intermediate_data_path.parent, exist_ok=True)

def fix_framenum_timestamp_mismatches(frames, fps, cam_jpg_dir, event_glob):
    frame_times = np.array([int(re.findall(timestamp_pattern, frame.stem)[0].split('currentTime_')[-1])*1e-9 
                            for frame in frames])
    frame_numbers  = np.array([int(re.findall(framenum_pattern, frame.stem)[0].split('frame_')[-1])
                                            for frame in frames])
    frame_diffs_from_timestamps = np.round(np.diff(frame_times) * fps).astype(int)
    frame_diffs_from_framenums  = np.diff(frame_numbers)
    
    mismatches = np.where(frame_diffs_from_timestamps > frame_diffs_from_framenums)[0]
    if len(mismatches) == 0:
        return frames
    else:
        for mismatch_idx in mismatches:
            correct_frame_diff = frame_diffs_from_timestamps[mismatch_idx]
            current_frame_diff = frame_diffs_from_framenums[mismatch_idx]
            framenum_addition = correct_frame_diff - current_frame_diff
            for frame in frames[mismatch_idx+1:]:
                filename = frame.name
                old_framenum = int(re.findall(framenum_pattern, filename)[0].split('frame_')[-1])
                new_framenum = str(old_framenum + framenum_addition).zfill(7)  
                
                new_filename = re.sub(framenum_pattern, f'frame_{new_framenum}', filename)
                new_frame = frame.parent / new_filename
                frame.rename(new_frame)
            
            frames = sorted(list(cam_jpg_dir.glob(event_glob)))
                
        return frames

def connect_event_fragments_to_previous_event(cam_list, event_idx_range_list, correct_event_idx_list, fps, jpg_dir):
    for cam_num, events_idx_range, correct_event_idx in zip(cam_list, event_idx_range_list, correct_event_idx_list):
        cam_jpg_dir = jpg_dir / f'jpg_cam{cam_num}'
        for event_idx in events_idx_range:
            event_num = event_idx + 1
    
            frames = sorted(list(cam_jpg_dir.glob(f'*event_{str(event_num).zfill(3)}_*')))
            if len(frames) == 0:
                continue
            last_correct_event_frame = sorted(list(cam_jpg_dir.glob(f'*event_{str(correct_event_idx+1).zfill(3)}_*')))[-1] 
            correct_start_time = re.findall(start_time_pattern, last_correct_event_frame.name)[0]
            last_frame_num     = int(re.findall(  framenum_pattern, last_correct_event_frame.stem)[0].split('frame_')[-1])
            last_timestamp     = int(re.findall( timestamp_pattern, last_correct_event_frame.stem)[0].split('currentTime_')[-1])
            
            current_event_first_timestamp = int(re.findall(timestamp_pattern, frames[0].stem)[0].split('currentTime_')[-1])
            inter_event_frame_count = np.round((current_event_first_timestamp - last_timestamp)*1e-9 *fps).astype(int)
                    
            for frame in frames:
                filename = frame.name
                old_framenum = int(re.findall(framenum_pattern, filename)[0].split('frame_')[-1])
                new_framenum = str(last_frame_num + inter_event_frame_count + old_framenum - 1).zfill(7)
                       
                new_filename = filename
                sub_pairs = [(event_pattern, f'event_{str(correct_event_idx+1).zfill(3)}_'),
                             (start_time_pattern, correct_start_time),
                             (framenum_pattern, f'frame_{new_framenum}')]
                for pair in sub_pairs:
                    new_filename = re.sub(pair[0], pair[1], new_filename)
                
                new_frame = frame.parent / new_filename
                frame.rename(new_frame)

def change_event_numbers_to_next_good_event(cam_list, event_idx_range_list, correct_event_idx_range_list, jpg_dir):
    for cam_num, events_idx_range, corrected_event_idxs in zip(cam_list, event_idx_range_list, correct_event_idx_range_list):
        cam_jpg_dir = jpg_dir / f'jpg_cam{cam_num}'
        for event_idx, correct_event_idx in zip(events_idx_range, corrected_event_idxs):
            event_num = event_idx + 1
            frames = sorted(list(cam_jpg_dir.glob(f'*event_{str(event_num).zfill(3)}_*')))
            for frame in frames:
                filename = frame.name
                new_filename = re.sub(event_pattern, f'event_{str(correct_event_idx+1).zfill(3)}_', filename)
                new_frame = frame.parent / new_filename
                frame.rename(new_frame)
 
def collect_frameNums_startTimes_lastFrames(jpg_dir, cam_list, fps): 
    lastEvents = []
    events_with_frames_list = []
    for cam_num in cam_list:
        cam_jpg_dir = jpg_dir / f'jpg_cam{cam_num}'
        all_frames = list(cam_jpg_dir.glob('*.jpg'))
        events_with_frames = np.unique([int(re.findall(event_pattern, fr.stem)[0].split('event_')[-1][:-1]) for fr in all_frames])
        lastEvents.append(max(events_with_frames))
        events_with_frames_list.append(events_with_frames)

    maxLastEvent = max(lastEvents)
    event_frame_nums  = np.zeros((maxLastEvent, len(cam_list)))
    event_last_frame  = np.full((maxLastEvent, len(cam_list)), np.nan)
    event_start_times = np.full((maxLastEvent, len(cam_list)), '-------', dtype="S7")
    event_length_sec  = np.zeros_like(event_frame_nums)
    for cIdx, (cam_num, events_list) in enumerate(zip(cam_list, events_with_frames_list)):
        for event_num in events_list:
            eIdx = event_num - 1
            cam_jpg_dir = jpg_dir / f'jpg_cam{cam_num}'
            event_glob = f'*event_{str(event_num).zfill(3)}_*'
            print(cam_jpg_dir / event_glob)
            
            frames = sorted(list(cam_jpg_dir.glob(event_glob)))            
            if len(frames) > 0:
                frames = fix_framenum_timestamp_mismatches(frames, fps, cam_jpg_dir, event_glob)
                event_frame_nums[eIdx, cIdx] = len(frames)
                event_last_frame[eIdx, cIdx] = int(re.findall(framenum_pattern, frames[-1].stem)[0].split('frame_')[-1])
                start_times = [re.findall(start_time_pattern, frame.name)[0].split('.jpg')[0] for frame in frames]
                if len(start_times) > 0:
                    if len(np.unique(start_times)) > 1:
                        print('multiple start times')
                    else:
                        event_start_times[eIdx, cIdx] = start_times[0]
                
                first_timestamp = int(re.findall(timestamp_pattern, frames[ 0].stem)[0].split('currentTime_')[-1])
                last_timestamp  = int(re.findall(timestamp_pattern, frames[-1].stem)[0].split('currentTime_')[-1])
                event_length_sec[eIdx, cIdx] = np.round((last_timestamp - first_timestamp) * 1e-9, 3)
    
    return event_frame_nums, event_start_times, event_last_frame, event_length_sec, maxLastEvent

def load_or_run_event_fragments(cam_list, event_idx_range_list, correct_event_idx_list, fps, jpg_dir, all_cams_list, key):
    
    try:
        inter_data = load_dict_from_hdf5(intermediate_data_path)
        event_collection_dict = inter_data[key]
    except:
        connect_event_fragments_to_previous_event(cam_list               = cam_list,
                                                  event_idx_range_list   = event_idx_range_list,
                                                  correct_event_idx_list = correct_event_idx_list,
                                                  fps                    = fps,
                                                  jpg_dir                = jpg_dir)
        
        event_collection = collect_frameNums_startTimes_lastFrames(jpg_dir, all_cams_list, fps)
    
        event_collection_dict= dict(event_frame_nums  = event_collection[0], 
                                    event_start_times = event_collection[1], 
                                    event_last_frame  = event_collection[2], 
                                    event_length_sec  = event_collection[3], 
                                    lastEvent         = event_collection[4])
        
        save_dict_to_hdf5(event_collection_dict, intermediate_data_path, first_level_key=key)
    
    event_collection_dict['event_start_times_objectType'] = event_collection_dict['event_start_times'].astype(object)    
    
    return event_collection_dict, inter_data

def load_or_run_change_event_nums(cam_list, event_idx_range_list, correct_event_idx_range_list, jpg_dir, all_cams_list, key):

    try:
        inter_data = load_dict_from_hdf5(intermediate_data_path)
        event_collection_dict = inter_data[key]
    except:     
        change_event_numbers_to_next_good_event(cam_list                     = cam_list, 
                                                event_idx_range_list         = event_idx_range_list, 
                                                correct_event_idx_range_list = correct_event_idx_range_list,
                                                jpg_dir                      = jpg_dir)
        
        event_collection = collect_frameNums_startTimes_lastFrames(jpg_dir, all_cams_list, fps)
        
        event_collection_dict= dict(event_frame_nums  = event_collection[0], 
                                    event_start_times = event_collection[1], 
                                    event_last_frame  = event_collection[2], 
                                    event_length_sec  = event_collection[3], 
                                    lastEvent         = event_collection[4])
        
        save_dict_to_hdf5(event_collection_dict, intermediate_data_path, first_level_key=key)
    
    event_collection_dict['event_start_times_objectType'] = event_collection_dict['event_start_times'].astype(object)    
    
    return event_collection_dict, inter_data

if __name__ == '__main__':

    # Load existing data or gather metadata from all files, then save. 
    try:
        inter_data = load_dict_from_hdf5(intermediate_data_path)
        event_dict = inter_data['original']
    except:
        event_collection = collect_frameNums_startTimes_lastFrames(jpg_dir, all_cams_list, fps)
        event_dict = dict(event_frame_nums  = event_collection[0], 
                                     event_start_times = event_collection[1], 
                                     event_last_frame  = event_collection[2], 
                                     event_length_sec  = event_collection[3], 
                                     lastEvent         = event_collection[4])
        save_dict_to_hdf5(event_dict, intermediate_data_path, first_level_key='original')
    event_dict['event_start_times_objectType'] = event_dict['event_start_times'].astype(object)
    
    
    if not first_run:

        key = 'fr'
        event_dict, inter_data = load_or_run_event_fragments(cam_list               = [          1,          2,          3,          4,          5], # FIXME
                                                             event_idx_range_list   = [range(1, 9),range(1, 6),range(1, 6),range(1,10), range(1,9)], # FIXME
                                                             correct_event_idx_list = [          0,          0,          0,          0,          0], # FIXME
                                                             fps                    = fps,
                                                             jpg_dir                = jpg_dir,
                                                             all_cams_list          = all_cams_list,
                                                             key                    = key) 
        
        key += '_ev'
        event_dict, inter_data = load_or_run_change_event_nums(cam_list                     = [           1,          2,          3,          4 ,          5], # FIXME
                                                               event_idx_range_list         = [range(9, 10),range(6, 7),range(6, 7),range(10,11),range(9,10)], # FIXME
                                                               correct_event_idx_range_list = [range(1,  2),range(1, 2),range(1, 2), range(1, 2),range(1, 2)], # FIXME
                                                               jpg_dir                      = jpg_dir,
                                                               all_cams_list                = all_cams_list,
                                                               key                          = key)
        
        key += '_fr'
        lastEvent = event_collection_dict['lastEvent']
        event_dict, inter_data = load_or_run_event_fragments(cam_list               = [            1,           2,           3,           4,           5], # FIXME
                                                             event_idx_range_list   = [range(10, 35),range(7, 28),range(7, 26),range(11,35),range(10,33)], # FIXME
                                                             correct_event_idx_list = [            1,           1,           1,           1,           1], # FIXME
                                                             fps                    = fps,
                                                             jpg_dir                = jpg_dir,
                                                             all_cams_list          = all_cams_list,
                                                             key                    = key) 
        
        key += '_ev'
        event_dict, inter_data = load_or_run_change_event_nums(cam_list                     = [           1,           2,           3,           4,           5], # FIXME
                                                               event_idx_range_list         = [range(35,36),range(28,29),range(26,27),range(35,36),range(33,34)], # FIXME
                                                               correct_event_idx_range_list = [range( 2, 3),range( 2, 3),range( 2, 3),range( 2, 3),range( 2, 3)], # FIXME
                                                               jpg_dir                      = jpg_dir,
                                                               all_cams_list                = all_cams_list,
                                                               key                          = key)
        
        key += '_fr'
        lastEvent = event_collection_dict['lastEvent']
        event_dict, inter_data = load_or_run_event_fragments(cam_list               = [           1,           2,           3,           4,           5], # FIXME
                                                             event_idx_range_list   = [range(36,81),range(29,66),range(27,73),range(36,71),range(34,70)], # FIXME
                                                             correct_event_idx_list = [           2,           2,           2,           2,           2], # FIXME
                                                             fps                    = fps,
                                                             jpg_dir                = jpg_dir,
                                                             all_cams_list          = all_cams_list,
                                                             key                    = key) 
        
        key += '_ev'
        event_dict, inter_data = load_or_run_change_event_nums(cam_list                     = [           1,           2,           3,           4,           5], # FIXME
                                                               event_idx_range_list         = [range(81,82),range(66,67),range(73,74),range(71,72),range(70,71)], # FIXME
                                                               correct_event_idx_range_list = [range( 3, 4),range( 3, 4),range( 3, 4),range( 3, 4),range( 3, 4)], # FIXME
                                                               jpg_dir                      = jpg_dir,
                                                               all_cams_list                = all_cams_list,
                                                               key                          = key)
        
        key += '_fr'
        lastEvent = event_collection_dict['lastEvent']
        event_dict, inter_data = load_or_run_event_fragments(cam_list               = [            1,            2,            3,            4,            5], # FIXME
                                                             event_idx_range_list   = [range(82,172),range(67,162),range(74,178),range(72,155),range(71,171)], # FIXME
                                                             correct_event_idx_list = [            3,            3,            3,            3,            3], # FIXME
                                                             fps                    = fps,
                                                             jpg_dir                = jpg_dir,
                                                             all_cams_list          = all_cams_list,
                                                             key                    = key) 
        
        key += '_ev'
        event_dict, inter_data = load_or_run_change_event_nums(cam_list                     = [             1,             2,             3,             4,             5], # FIXME
                                                               event_idx_range_list         = [range(172,173),range(162,163),range(178,179),range(155,156),range(171,172)], # FIXME
                                                               correct_event_idx_range_list = [range(  4,  5),range(  4,  5),range(  4,  5),range(  4,  5),range(  4,  5)], # FIXME
                                                               jpg_dir                      = jpg_dir,
                                                               all_cams_list                = all_cams_list,
                                                               key                          = key)

        key += '_fr'
        lastEvent = event_collection_dict['lastEvent']
        event_dict, inter_data = load_or_run_event_fragments(cam_list               = [             1,             2,             3,             4,             5], # FIXME
                                                             event_idx_range_list   = [range(173,279),range(163,278),range(179,286),range(156,257),range(172,276)], # FIXME
                                                             correct_event_idx_list = [             4,             4,             4,             4,             4], # FIXME
                                                             fps                    = fps,
                                                             jpg_dir                = jpg_dir,
                                                             all_cams_list          = all_cams_list,
                                                             key                    = key)    

        key += '_ev'
        event_dict, inter_data = load_or_run_change_event_nums(cam_list                     = [             1,             2,             3,             4,             5], # FIXME
                                                               event_idx_range_list         = [range(279,280),range(278,279),range(286,287),range(257,258),range(276,277)], # FIXME
                                                               correct_event_idx_range_list = [range(  5,  6),range(  5,  6),range(  5,  6),range(  5,  6),range(  5,  6)], # FIXME
                                                               jpg_dir                      = jpg_dir,
                                                               all_cams_list                = all_cams_list,
                                                               key                          = key)    
        
        key += '_fr'
        lastEvent = event_collection_dict['lastEvent']
        event_dict, inter_data = load_or_run_event_fragments(cam_list               = [             1,             2,             3,             4,             5], # FIXME
                                                             event_idx_range_list   = [range(280,331),range(279,331),range(287,331),range(258,331),range(277,331)], # FIXME
                                                             correct_event_idx_list = [             5,             5,             5,             5,             5], # FIXME
                                                             fps                    = fps,
                                                             jpg_dir                = jpg_dir,
                                                             all_cams_list          = all_cams_list,
                                                             key                    = key)    