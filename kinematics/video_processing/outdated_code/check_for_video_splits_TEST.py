#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:17:06 2022

@author: daltonm
"""

import os
import glob
import numpy as np
import pandas as pd
import re

dates = ['2022_10_24']
exp = 'test'
marms = 'TYJL'
vid_dir = '/project/nicho/data/marmosets/kinematics_videos'
jpg_dir = '/scratch/midway3/daltonm/kinematics_jpgs'
session_nums = [1, 2]
ncams = 5
fps = 200

event_pattern      = re.compile('event_\d{3}')
start_time_pattern = re.compile('\d{4}-\d{2}.jpg')
timestamp_pattern  = re.compile('currentTime_\d{9,20}')
framenum_pattern   = re.compile('frame_\d{7}')


for date in dates:
    split_video_record_path = os.path.join(vid_dir, '%s/%s/%s/split_video_records' % (exp, marms, date))
    os.makedirs(split_video_record_path, exist_ok=True)

    # collect all session_event_cam combinations
    vid_frame_nums = [0]*len(session_nums)
    vid_start_times = [0]*len(session_nums)
    for sIdx, sNum in enumerate(session_nums):
        jpg_path = os.path.join(jpg_dir, '%s/%s/%s/session%d' % (exp, marms, date, sNum))
        
        lastEvents = []
        for cam in range(1, ncams+1):

            jpg_file = sorted(glob.glob(os.path.join(jpg_path, 'jpg_cam%d' % cam, '*')))[-1]
            lastEvents.append(int(re.findall(event_pattern, jpg_file)[0].split('event_')[-1]))

        lastEvent = max(lastEvents)
        event_frame_nums = np.zeros((lastEvent, ncams))
        event_start_times = np.full_like(event_frame_nums, np.nan, dtype=object)
        for cIdx, cam in enumerate(range(1, ncams+1)):
            for eIdx, eNum in enumerate(range(1, lastEvent+1)):
                print(cam, eNum)
                event=str(eNum).zfill(3)
                cam_img_path = os.path.join(jpg_path, 'jpg_cam%d' % cam)
                event_image_files = sorted(glob.glob(os.path.join(cam_img_path, '*cam%d_event_%s*' % (cam, event))))
                event_frame_nums[eIdx, cIdx] = len(event_image_files)
                start_times = [re.findall(start_time_pattern, tmp_image_file)[0].split('.jpg')[0] for tmp_image_file in event_image_files]
                if len(start_times) > 0:
                    if len(np.unique(start_times)) > 1:
                        print('multiple start times')
                    else:
                        event_start_times[eIdx, cIdx] = start_times[0]
                        
        event_start_times_original = event_start_times.copy()
        event_frame_nums_original  = event_frame_nums.copy()
        for eIdx in range(lastEvent-1, -1, -1):
            cam_event_time = [val for val in event_start_times[eIdx] if type(val) == str]
            if len(cam_event_time) < ncams:
                if len(np.unique(cam_event_time)) == 1:
                    event_time = cam_event_time[0]
                    for cIdx in range(event_start_times.shape[1]):
                        current_event_idx = [idx for idx, eTime in enumerate(event_start_times[:, cIdx]) if eTime == event_time]
                        if len(current_event_idx) == 1:
                            if current_event_idx[0] == eIdx:
                                continue
                            current_event_idx = current_event_idx[0]
                            values_between = event_start_times[current_event_idx+1 : eIdx+1, cIdx]
                            start_time_idxs = [idx for idx, val in enumerate(values_between) if type(val) == str]
                            num_nan_times = len([val for val in values_between if type(val) == float])
                            if num_nan_times == len(values_between):
                                event_start_times[eIdx, cIdx] = event_time
                                event_start_times[current_event_idx, cIdx] = np.nan
                                event_frame_nums [eIdx, cIdx] = event_frame_nums[current_event_idx, cIdx]
                                event_frame_nums [current_event_idx, cIdx] = np.nan
                            else:
                                rows_to_add = start_time_idxs[-1] + (lastEvent - event_frame_nums.shape[0])
                                event_start_times = np.concatenate((event_start_times, 
                                                                    np.full((rows_to_add, event_frame_nums.shape[1]), np.nan, dtype=object)),
                                                                   axis = 0)
                                event_start_times[eIdx, cIdx] = event_time
                                event_start_times[eIdx+1 : start_time_idxs[-1]+1, cIdx] = values_between[:start_time_idxs[-1]+1]
                                event_start_times[current_event_idx : eIdx, cIdx] = np.nan
                                print('\n\n There are videos starting after the last video in the adjusted camera %d. LOOK INTO THIS! \n\n' % cIdx+1, flush=True)
                        elif len(current_event_idx) > 1:
                            print('\n\n Multiple events have this starting time. LOOK INTO THIS \n\n', flush=True)
                        # else:
                        #     print('\n\n No events have this starting time. moving to next \n\n', flush=True)

                else:
                    print('\n\n multiple different last event start times. LOOK INTO THIS!\n\n', flush=True)
                
        if np.all(event_frame_nums == event_frame_nums_original):
            print('breaking out of session %d' % sIdx)
            continue
        
        frame_nums_df  = pd.DataFrame(data = event_frame_nums , columns = ['cam%d' % cam for cam in range(1, ncams+1)])
        start_times_df = pd.DataFrame(data = event_start_times, columns = ['cam%d' % cam for cam in range(1, ncams+1)])
                
        vid_frame_nums [sIdx] = frame_nums_df
        vid_start_times[sIdx] = start_times_df
        
        first_timestamps = [0]*ncams
        cam_timestamps = [0]*ncams
        cam_event_image_files = [0]*ncams
        for cIdx, cam in enumerate(range(1, ncams+1)):
            cam_img_path = os.path.join(jpg_path, 'jpg_cam%d' % cam)
            
            adjusted_event_idxs = [idx for idx, val in enumerate(event_start_times[:, cIdx]) if type(val) == str] 
            tmp_timestamps  = [0]*event_start_times.shape[0]
            tmp_image_files = [0]*event_start_times.shape[0]
            for eIdx, adjusted_idx in enumerate(adjusted_event_idxs):
            #for eIdx, eNum in enumerate(range(1, lastEvent+1)):
                eNum = eIdx+1
                print(cam, eNum)
                event=str(eNum).zfill(3)
                tmp_image_files[adjusted_idx] = sorted(glob.glob(os.path.join(cam_img_path, '*cam%d_event_%s*' % (cam, event))))
                if eIdx == 0:
                    first_timestamps[cIdx] = int(re.findall(timestamp_pattern, tmp_image_files[adjusted_idx][0])[0].split('_')[-1])
                # tmp_timestamps[eIdx]= [int(re.findall(timestamp_pattern, img)[0]) - first_timestamps[cIdx] for img in event_image_files]         
                tmp_timestamps[adjusted_idx] = [int(re.findall(timestamp_pattern, img)[0].split('_')[-1]) for img in tmp_image_files[adjusted_idx]]
            
            cam_timestamps[cIdx]        = tmp_timestamps
            cam_event_image_files[cIdx] = tmp_image_files 
        
        print('changing filenames to stitch videos togetther')
        for cIdx, (timestamps, image_files) in enumerate(zip(cam_timestamps, cam_event_image_files)):
            for eIdx in range(1, len(timestamps)):
                
                prev_eIdx = np.where(~np.isnan(event_frame_nums[:eIdx, cIdx]))[0][-1]
                
                event_timestamps = timestamps[eIdx]
                prev_timestamps  = timestamps[prev_eIdx]
                start_time       = event_start_times[eIdx, cIdx]
                
                event_files      = image_files[eIdx]
                prev_event_files = image_files[prev_eIdx] 
                
                all_cams_start_times = event_start_times[eIdx]
                if type(start_time) == str and not all([start_time == start for start in all_cams_start_times]):
                    frame_period_prev = np.round(np.mean(np.diff(prev_timestamps))*1e-6).astype(int) # in ms

                    inter_event_frame_count = np.round((event_timestamps[0] - prev_timestamps[-1])*1e-6 / frame_period_prev).astype(int)
                    if inter_event_frame_count < fps/4:
                        
                        prev_epi_last_file = prev_event_files[-1]
                        prev_epi_num             = re.findall(event_pattern,        prev_epi_last_file)[0]
                        prev_epi_start_time      = re.findall(start_time_pattern,   prev_epi_last_file)[0]
                        prev_epi_last_framenum   = int(re.findall(framenum_pattern, prev_epi_last_file)[0].split('frame_')[-1])
                        
                        for img_file in event_files:
                            
                            img_framenum = int(re.findall(framenum_pattern, img_file)[0].split('frame_')[-1])
                            new_framenum = str(prev_epi_last_framenum + inter_event_frame_count + img_framenum - 1).zfill(7)
                            
                            new_img_file = img_file
                            sub_pairs = [(event_pattern, prev_epi_num),
                                         (start_time_pattern, prev_epi_start_time),
                                         (framenum_pattern, 'frame_%s' % new_framenum)]
                            for pair in sub_pairs:
                                new_img_file = re.sub(pair[0], pair[1], new_img_file)
                            
                            os.rename(img_file, new_img_file)
                                                
                        event_frame_nums [prev_eIdx, cIdx] = event_frame_nums[prev_eIdx, cIdx] + len(event_timestamps)
                        event_frame_nums [eIdx, cIdx]      = np.nan
                        event_start_times[eIdx, cIdx]      = np.nan

                        timestamps[prev_eIdx].extend(event_timestamps)
                        timestamps[eIdx] = 0
                        
        
        event_frame_nums[event_frame_nums == 0] = np.nan
        nan_rows = np.where(np.isnan(event_frame_nums))[0]
        empty_episodes = [row for row in np.unique(nan_rows) if np.sum(nan_rows == row) == ncams]
        
        empty_diff_tmp = np.insert(np.diff(empty_episodes), 0, 5)
        empty_start = [epi for epi, diff in zip(empty_episodes, empty_diff_tmp) if diff > 1]
        empty_diff_tmp = np.append(np.diff(empty_episodes), 5)
        empty_end   = [epi for epi, diff in zip(empty_episodes, empty_diff_tmp) if diff > 1]
        
        empty_episode_ranges = [range(start, end+1) for start, end in zip(empty_start, empty_end)]
        empty_end.append(event_frame_nums.shape[0])
        empty_start.append(event_frame_nums.shape[0]-1)
        episode_ranges_to_edit = [(end+1, start) for start, end in zip(empty_start[1:], empty_end[:-1])]
        
        print('starting to change event nums on last set of videos')
        for cIdx, (timestamps, image_files) in enumerate(zip(cam_timestamps, cam_event_image_files)):
            for empty_range, epi_range in zip(empty_episode_ranges, episode_ranges_to_edit):
                for new_epi_idx, start_time, epi_images in zip(empty_range, 
                                                               event_start_times[epi_range[0] : epi_range[1]+1, cIdx],
                                                               image_files[epi_range[0] : epi_range[1]+1]):
                    new_epi_num = str(new_epi_idx + 1).zfill(3)
                    old_epi_num = str(np.where(event_start_times_original[:, cIdx] == start_time)[0][-1] + 1).zfill(3)
                    
                    if re.findall(event_pattern, epi_images[0])[0].split('event_')[-1] == old_epi_num:
                        for img_file in epi_images:
                            new_img_file = img_file
                            sub_pairs = [(event_pattern, 'event_%s' % new_epi_num)]
                            for pair in sub_pairs:
                                new_img_file = re.sub(pair[0], pair[1], new_img_file)
                                os.rename(img_file, new_img_file)
                    else:
                        print('Episode number in the image filenames does not match the expected episode number')
                                           