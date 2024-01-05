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

session=2
date='2023_11_24'
exp = 'foraging'
marms = 'JLTY'
intermediate_data_path = Path(f'/project/nicho/data/marmosets/kinematics_videos/{exp}/{marms}/{date}/manual_corrections_records/session{session}_episode_correction_record.h5')
jpg_dir = Path(f'/scratch/midway3/daltonm/kinematics_jpgs/{exp}/{marms}/{date}/session{session}/')
fps = 200
all_cams_list = [1, 2, 3, 4, 5]


event_pattern      = re.compile('event_\d{3}')
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
    
            frames = sorted(list(cam_jpg_dir.glob(f'*event_{str(event_num).zfill(3)}*')))
            if len(frames) == 0:
                continue
            last_correct_event_frame = sorted(list(cam_jpg_dir.glob(f'*event_{str(correct_event_idx+1).zfill(3)}*')))[-1] 
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
                sub_pairs = [(event_pattern, f'event_{str(correct_event_idx+1).zfill(3)}'),
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
            frames = sorted(list(cam_jpg_dir.glob(f'*event_{str(event_num).zfill(3)}*')))
            for frame in frames:
                filename = frame.name
                new_filename = re.sub(event_pattern, f'event_{str(correct_event_idx+1).zfill(3)}', filename)
                new_frame = frame.parent / new_filename
                frame.rename(new_frame)
 
def collect_frameNums_startTimes_lastFrames(jpg_dir, cam_list, fps): 
    lastEvents = []
    for cam_num in cam_list:
        cam_jpg_dir = jpg_dir / f'jpg_cam{cam_num}'
        last_frame = sorted(list(cam_jpg_dir.glob('*.jpg')))[-1].stem
        lastEvents.append(int(re.findall(event_pattern, last_frame)[0].split('event_')[-1]))

    lastEvent = max(lastEvents)
    event_frame_nums  = np.zeros((lastEvent, len(cam_list)))
    event_last_frame  = np.full((lastEvent, len(cam_list)), np.nan)
    event_start_times = np.full((lastEvent, len(cam_list)), '-------', dtype="S7")
    event_length_sec  = np.zeros_like(event_frame_nums)
    for cIdx, cam_num in enumerate(cam_list):
        for eIdx, event_num in enumerate(range(1, lastEvent+1)):
            cam_jpg_dir = jpg_dir / f'jpg_cam{cam_num}'
            event_glob = f'*event_{str(event_num).zfill(3)}*'
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
    
    return event_frame_nums, event_start_times, event_last_frame, event_length_sec, lastEvent

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
        event_collection_dict = inter_data['original']
    except:
        event_collection = collect_frameNums_startTimes_lastFrames(jpg_dir, [1, 2, 3, 4, 5], fps)
        event_collection_dict = dict(event_frame_nums  = event_collection[0], 
                                     event_start_times = event_collection[1], 
                                     event_last_frame  = event_collection[2], 
                                     event_length_sec  = event_collection[3], 
                                     lastEvent         = event_collection[4])
        save_dict_to_hdf5(event_collection_dict, intermediate_data_path, first_level_key='original')
    event_collection_dict['event_start_times_objectType'] = event_collection_dict['event_start_times'].astype(object)
    
    
    if not first_run:
        '''For each camera that needs event fragments tacked on to the end of the a previous event:
            - Add the cam_num (not index) to cam_list.   
            - Set the corresponding element of event_idx_range_list to range(firstFragment_idx, nextGoodEvent_idx).
            - Set the corresponding element of correct_event_idx_list to prevGoodEvent_idx.
            - check that <key> is set to something sensible. I have been documenting each step by adding an underscore and the type of the step at each step.
            
            prevGoodEvent_idx: found by identifying the event that started at the same time as a "good" reference camera event, but eventually gets fragmented.
            firstFragment_idx: the next index after prevGoodEvent_idx.
            nextGoodEvent_idx: the index at which the event_start_time matches the reference event_start_time in a "good" camera. 
                               Typically, the frame count here also matches the frame count for the good reference camera.
        
        If the data is already stored in the h5 file, the function will load it and skip this step.
        '''
        key = 'fragments'
        event_dict, inter_data = load_or_run_event_fragments(cam_list               = [            3,             4,            5], # FIXME
                                                             event_idx_range_list   = [range(8, 178), range(8, 156), range(8, 41)], # FIXME
                                                             correct_event_idx_list = [           7,              7,            7], # FIXME
                                                             fps                    = fps,
                                                             jpg_dir                = jpg_dir,
                                                             all_cams_list          = all_cams_list,
                                                             key                    = key) 
        
        '''For each camera that needs to change a set of events to the now-corrected event number:
            - Add the cam_num (not index) to cam_list.   
            - Set the corresponding element of event_idx_range_list to range(nextGoodEvent_idx, nextFragment_idx).
            - Set the corresponding element of correct_event_idx_range_list to 
                        range(prevGoodEvent_idx +1, prevGoodEvent_idx +1 + (nextFragment_idx-nextGoodEvent_idx)).
            - check that <key> is set to something sensible. I have been documenting each step by adding an underscore and the type of the step at each step.
            
            nextGoodEvent_idx: Taken from the fragments correction step.
            nextFragment_idx : basically, for the next block of events you should identify the last event that started at the same time as the reference good camera event.
                               Then add 1, so that the range changes everything up through the beginning of the event that will eventually fragment. (same logic as firstFragment_idx above)
            firstFragment_idx: Taken from the fragments correction step.
            prevGoodEvent_idx: Taken from the fragments correction step.

        IMPORTANT: Make sure when setting the event_idx_range_list, you ***ADD 1*** at the end of the range. 
                   It will be natural to set it to stop at the last event idx you want to correct, but just remember
                   that python range variables run up to but do not include the "stop" index.
        
        If the data is already stored in the h5 file, the function will load it and skip this step.
        '''
        key += '_eventNum'
        event_dict, inter_data = load_or_run_change_event_nums(cam_list                     = [             3,                4,             5], # FIXME
                                                               event_idx_range_list         = [range(178, 188), range(156, 166), range(41, 64)], # FIXME
                                                               correct_event_idx_range_list = [range(  8,  18), range(  8,  18), range( 8, 31)], # FIXME
                                                               jpg_dir                      = jpg_dir,
                                                               all_cams_list                = all_cams_list,
                                                               key                          = key)

        '''
            Same logic described for previous fragments correction. 
            The correct_event_idx_list element = 1 less than the stop value of the previous correct_event_idx_range_list.
        '''
        key += '_fragments'
        lastEvent = event_collection_dict['lastEvent']
        event_dict, inter_data = load_or_run_event_fragments(cam_list               = [              3,               4], # FIXME
                                                             event_idx_range_list   = [range(188, 228), range(166, 229)], # FIXME
                                                             correct_event_idx_list = [             17,              17], # FIXME
                                                             fps                    = fps,
                                                             jpg_dir                = jpg_dir,
                                                             all_cams_list          = all_cams_list,
                                                             key                    = key) 
        
        '''
            Same logic described for previous eventNum correction.
        '''
        key += '_eventNum'
        event_dict, inter_data = load_or_run_change_event_nums(cam_list                     = [             3,                4], # FIXME
                                                               event_idx_range_list         = [range(228, 241), range(229, 242)], # FIXME
                                                               correct_event_idx_range_list = [range( 18,  31), range( 18,  31)], # FIXME
                                                               jpg_dir                      = jpg_dir,
                                                               all_cams_list                = all_cams_list,
                                                               key                          = key)