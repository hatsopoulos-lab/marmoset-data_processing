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

intermediate_data_path = '/project/nicho/data/marmosets/kinematics_videos/foraging/JLTY/2023_11_25/metadata_from_kinematics_processing//INTERMEDIATE_episode_correction_record.pkl'
jpg_dir = Path('/scratch/midway3/daltonm/kinematics_jpgs/foraging/JLTY/2023_11_25/session1/')
fps = 200

# with open(intermediate_data_path, 'rb') as f:
#     inter_data = dill.load(f)
#     event_frame_nums = inter_data['event_frame_nums']
#     event_start_times = inter_data['event_start_times']
#     lastEvent = inter_data['lastEvent']

event_pattern      = re.compile('event_\d{3}')
start_time_pattern = re.compile('\d{4}-\d{2}.jpg')
timestamp_pattern  = re.compile('currentTime_\d{9,20}')
framenum_pattern   = re.compile('frame_\d{7}')

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

# correct event numbers, frame numbers, and the event_start_time for events after event 001 that SHOULD be part of event 001
def connect_event_fragments_to_previous_event(cam_list, event_range_list, correct_event_list, fps, jpg_dir):
    for cam_num, events_idx_range, correct_event in zip(cam_list, event_range_list, correct_event_list):
        cam_jpg_dir = jpg_dir / f'jpg_cam{cam_num}'
        for event_idx in events_idx_range:
            event_num = event_idx + 1
    
            frames = sorted(list(cam_jpg_dir.glob(f'*event_{str(event_num).zfill(3)}*')))
            if len(frames) == 0:
                continue
            last_correct_event_frame = sorted(list(cam_jpg_dir.glob(f'*event_{str(correct_event).zfill(3)}*')))[-1] 
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
                sub_pairs = [(event_pattern, f'event_{str(correct_event).zfill(3)}'),
                             (start_time_pattern, correct_start_time),
                             (framenum_pattern, f'frame_{new_framenum}')]
                for pair in sub_pairs:
                    new_filename = re.sub(pair[0], pair[1], new_filename)
                
                new_frame = frame.parent / new_filename
                frame.rename(new_frame)

# change event number for all the events that should be part of events 002, 003, and 004
def change_event_numbers_to_next_good_event(cam_list, event_range_list, correct_event_range_list, jpg_dir):
    for cam_num, events_idx_range, corrected_event_nums in zip(cam_list, event_range_list, correct_event_range_list):
        cam_jpg_dir = jpg_dir / f'jpg_cam{cam_num}'
        for event_idx, correct_event in zip(events_idx_range, corrected_event_nums):
            event_num = event_idx + 1
            frames = sorted(list(cam_jpg_dir.glob(f'*event_{str(event_num).zfill(3)}*')))
            for frame in frames:
                filename = frame.name
                new_filename = re.sub(event_pattern, f'event_{str(correct_event).zfill(3)}', filename)
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
    event_start_times = np.full_like(event_frame_nums, np.nan, dtype=object)
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

event_collection = collect_frameNums_startTimes_lastFrames(jpg_dir, [1, 2, 3, 4, 5], fps)
event_frame_nums, event_start_times, event_last_frame, event_length_sec, lastEvent = event_collection

connect_event_fragments_to_previous_event(cam_list           = [           3,            4,            5],
                                          event_range_list   = [range(1, 46), range(1, 50), range(1, 11)],
                                          correct_event_list = [           1,            1,            1],
                                          fps                = fps,
                                          jpg_dir            = jpg_dir)

corrected_event_collection = collect_frameNums_startTimes_lastFrames(jpg_dir, [1, 2, 3, 4, 5], fps)
corrected_event_frame_nums, corrected_event_start_times, corrected_event_last_frame, corrected_event_length_sec, corrected_lastEvent = corrected_event_collection

            
change_event_numbers_to_next_good_event(cam_list                 = [            3,             4,             5], 
                                        event_range_list         = [range(46, 49), range(50, 53), range(11, 14)], 
                                        correct_event_range_list = [range( 2,  5), range( 2,  5), range( 2,  5)],
                                        jpg_dir                  = jpg_dir)

corrected_event_collection = collect_frameNums_startTimes_lastFrames(jpg_dir, [1, 2, 3, 4, 5], fps)
corrected_event_frame_nums, corrected_event_start_times, corrected_event_last_frame, corrected_event_length_sec, corrected_lastEvent = corrected_event_collection


connect_event_fragments_to_previous_event(cam_list           = [                   3,                    4,                    5],
                                          event_range_list   = [range(49, lastEvent), range(53, lastEvent), range(14, lastEvent)],
                                          correct_event_list = [                   4,                    4,                    4],
                                          fps                = fps,
                                          jpg_dir            = jpg_dir)  

corrected_event_collection = collect_frameNums_startTimes_lastFrames(jpg_dir, [1, 2, 3, 4, 5], fps)
corrected_event_frame_nums, corrected_event_start_times, corrected_event_last_frame, corrected_event_length_sec, corrected_lastEvent = corrected_event_collection

        