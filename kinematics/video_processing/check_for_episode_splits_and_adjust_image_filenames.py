#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:17:06 2022

@author: daltonm
"""

import os
import glob
import numpy as np
import re
import argparse
import pandas as pd
import dill
import time
from pathlib import Path

# regex_patterns = {'event'      : re.compile('event_\d{3}'),
#                   'start_time' : re.compile('\d{4}-\d{2}.jpg'),
#                   'timestamp'  : re.compile('currentTime_\d{9,20}'),
#                   'framenum'   : re.compile('frame_\d{7}')}
event_pattern      = re.compile('event_\d{3}')
start_time_pattern = re.compile('\d{4}-\d{2}.jpg')
timestamp_pattern  = re.compile('currentTime_\d{9,20}')
framenum_pattern   = re.compile('frame_\d{7}')

def collect_episode_start_times_and_frame_counts(jpg_path, ncams): 
    lastEvents = []
    for cam in range(1, ncams+1):

        print(os.path.join(jpg_path, 'jpg_cam%d' % cam, '*'))
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
            print('\n\n')
            print(os.path.join(cam_img_path, '*cam%d_event_%s*' % (cam, event)))
            print('\n\n')
            event_image_files = sorted(glob.glob(os.path.join(cam_img_path, '*cam%d_event_%s*' % (cam, event))))
            event_frame_nums[eIdx, cIdx] = len(event_image_files)
            start_times = [re.findall(start_time_pattern, tmp_image_file)[0].split('.jpg')[0] for tmp_image_file in event_image_files]
            if len(start_times) > 0:
                if len(np.unique(start_times)) > 1:
                    print('multiple start times')
                else:
                    event_start_times[eIdx, cIdx] = start_times[0]
                    
            for jpg_file in event_image_files:
                if 'sleep' in os.path.basename(jpg_file) and 'session' not in os.path.basename(jpg_file):
                    jpg_file_new = jpg_file.replace('sleep_1', 'sleep_session_1')
                    os.rename(jpg_file, jpg_file_new)
                    jpg_file = jpg_file_new 
    
    return event_frame_nums, event_start_times, lastEvent

def adjust_episodes_to_align_matches(event_frame_nums, event_start_times, lastEvent, ncams, jpg_path):

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
                            print('\n\n There are videos starting after the last video in the adjusted camera %d. LOOK INTO THIS! \n\n' % (cIdx+1), flush=True)
                    elif len(current_event_idx) > 1:
                        print('\n\n Multiple events have this starting time. LOOK INTO THIS \n\n', flush=True)
                    else:
                        tmp_current_event_idx = [idx for idx, eTime in enumerate(event_start_times[:, cIdx]) 
                                                 if type(eTime) == str 
                                                 and abs(int(eTime.replace('-', '')) - int(event_time.replace('-', ''))) <= 1]
                        if len(tmp_current_event_idx) == 1:
                            if tmp_current_event_idx[0] == eIdx:
                                continue
                            current_event_idx = tmp_current_event_idx[0]
                            close_event_time = event_start_times[current_event_idx, cIdx]
                            values_between = event_start_times[current_event_idx+1 : eIdx+1, cIdx]
                            start_time_idxs = [idx for idx, val in enumerate(values_between) if type(val) == str]
                            num_nan_times = len([val for val in values_between if type(val) == float])
                            if num_nan_times == len(values_between):
                                event_start_times[eIdx, cIdx] = close_event_time
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

                        # print('\n\n No events have this starting time. moving to next \n\n', flush=True)

            else:
                print('\n\n multiple different last event start times. LOOK INTO THIS!\n\n', flush=True)
                      
    # TODO                  
    # remove events if present in only 1 camera, at end of session, and brief
    for eIdx in range(lastEvent):
        if event_frame_nums[eIdx].ptp() == 0:
            lastGoodEventIdx = eIdx        
    
    for eIdx in range(lastGoodEventIdx, lastEvent): 
        num_nan_event_times = len([val for val in event_start_times[eIdx] if type(val) == float])
        max_frame_nums = event_frame_nums[eIdx].max()
        if (len(event_start_times[eIdx]) - num_nan_event_times) == 1 and max_frame_nums < 100:
            cam = [cIdx for cIdx, val in enumerate(event_start_times[eIdx]) if type(val) == str][0]+1
            event_start_times[eIdx] = np.nan        
            event_frame_nums[eIdx] = 0
            
            event_files = list((Path(jpg_path) / f'jpg_cam{cam}').glob(f'*event_{str(eIdx+1).zfill(3)}*'))            
            for img_file in event_files:                      
                img_file.unlink()
    return event_frame_nums, event_start_times

def collect_timestamps_and_filenames(jpg_path, ncams, event_start_times):
    cam_timestamps = [0]*ncams
    cam_event_image_files = [0]*ncams
    for cIdx, cam in enumerate(range(1, ncams+1)):
        cam_img_path = os.path.join(jpg_path, 'jpg_cam%d' % cam)
        
        adjusted_event_idxs = [idx for idx, val in enumerate(event_start_times[:, cIdx]) if type(val) == str] 
        tmp_timestamps  = [0]*event_start_times.shape[0]
        tmp_image_files = [0]*event_start_times.shape[0]
        for eIdx, adjusted_idx in enumerate(adjusted_event_idxs):
            eNum = eIdx+1
            event=str(eNum).zfill(3)
            tmp_image_files[adjusted_idx] = sorted(glob.glob(os.path.join(cam_img_path, '*cam%d_event_%s*' % (cam, event))))       
            tmp_timestamps[adjusted_idx] = [int(re.findall(timestamp_pattern, img)[0].split('_')[-1]) for img in tmp_image_files[adjusted_idx]]
        
        cam_timestamps[cIdx]        = tmp_timestamps
        cam_event_image_files[cIdx] = tmp_image_files
    
    return cam_timestamps, cam_event_image_files

def stitch_episodes_together(cam_timestamps, cam_event_image_files, event_frame_nums, event_start_times, fps):
    for cIdx, (timestamps, image_files) in enumerate(zip(cam_timestamps, cam_event_image_files)):
        for eIdx in range(1, len(timestamps)):
            
            prev_eIdx = np.where(~np.isnan(event_frame_nums[:eIdx, cIdx]))[0][-1]
            
            event_timestamps = timestamps[eIdx]
            prev_timestamps  = timestamps[prev_eIdx]
            start_time       = event_start_times[eIdx, cIdx]
            
            event_files      = image_files[eIdx]
            prev_event_files = image_files[prev_eIdx] 
            
            all_cams_start_times = event_start_times[eIdx]
            prevIdx_all_cams_start_times = event_start_times[eIdx-1]
            if eIdx+1 < len(timestamps):
                nextIdx_all_cams_start_times = event_start_times[eIdx+1]
                
            if type(start_time) == str and not all([start_time == start for start in all_cams_start_times]):
                currentCheck = all([type(eTime) == str 
                                    and abs(int(eTime.replace('-', '')) - int(start_time.replace('-', ''))) <= 1 
                                    for eTime in all_cams_start_times])
                prevCheck    = all([type(eTime) == str and type(prevIdx_all_cams_start_times[0]) == str 
                                    and abs(int(eTime.replace('-', '')) - int(prevIdx_all_cams_start_times[0].replace('-', ''))) <= 1 
                                    for eTime in prevIdx_all_cams_start_times])
                if eIdx+1 < len(timestamps):
                    nextCheck    = all([type(eTime) == str and type(nextIdx_all_cams_start_times[0]) == str
                                        and abs(int(eTime.replace('-', '')) - int(nextIdx_all_cams_start_times[0].replace('-', ''))) <= 1 
                                        for eTime in nextIdx_all_cams_start_times])
                else:
                    nextCheck = True
                    
                if currentCheck and prevCheck and nextCheck:
                    print(eIdx)
                    continue
                
                frame_period_prev = np.round(np.mean(np.diff(prev_timestamps))*1e-9).astype(int) # in sec

                inter_event_frame_count = np.round((event_timestamps[0] - prev_timestamps[-1])*1e-9 / frame_period_prev).astype(int)
                if (type(prevIdx_all_cams_start_times[cIdx]) != str 
                    and np.isnan(prevIdx_all_cams_start_times[cIdx])
                    and any([start_time == start for start in prevIdx_all_cams_start_times])):  
                    
                    current_epi_last_file = event_files[-1]
                    current_epi_num       = re.findall(event_pattern,        current_epi_last_file)[0]
                    new_epi_num           = current_epi_num[:-3] + str(int(current_epi_num.split('_')[-1])-1).zfill(3)

                    print('\nChanging episode %s files to episode %s\n' % (current_epi_num.split('_')[-1],
                                                                           new_epi_num.split('_')[-1]), flush=True)
                    
                    for img_file in event_files:
                    
                        new_img_file = img_file
                        sub_pairs = [(event_pattern, new_epi_num)]
                        for pair in sub_pairs:
                            new_img_file = re.sub(pair[0], pair[1], new_img_file)

                        if not troubleshoot:                        
                            os.rename(img_file, new_img_file)
                        image_files[prev_eIdx].append(new_img_file)
                    
                    image_files[eIdx] = 0
                                            
                    event_frame_nums [eIdx-1, cIdx] = event_frame_nums[eIdx, cIdx]
                    event_frame_nums [eIdx  , cIdx] = np.nan
                    event_start_times[eIdx-1, cIdx] = event_start_times[eIdx, cIdx]
                    event_start_times[eIdx  , cIdx] = np.nan

                    timestamps[eIdx-1] = event_timestamps
                    timestamps[eIdx] = 0
                                                                 
                elif inter_event_frame_count < fps/4:
                    
                    prev_epi_last_file      = prev_event_files[-1]
                    prev_epi_num            = re.findall(event_pattern,        prev_epi_last_file)[0]
                    prev_epi_start_time     = re.findall(start_time_pattern,   prev_epi_last_file)[0]
                    prev_epi_last_framenum  = int(re.findall(framenum_pattern, prev_epi_last_file)[0].split('frame_')[-1])
                    
                    print('\nStitching image files %s thru %s onto the end of episode %s\n' % (os.path.basename(event_files[0]), 
                                                                                               os.path.basename(event_files[-1]), 
                                                                                               prev_epi_num.split('_')[-1]), flush=True)
                    
                    for img_file in event_files:
                        
                        img_framenum = int(re.findall(framenum_pattern, img_file)[0].split('frame_')[-1])
                        new_framenum = str(prev_epi_last_framenum + inter_event_frame_count + img_framenum - 1).zfill(7)
                        
                        new_img_file = img_file
                        sub_pairs = [(event_pattern, prev_epi_num),
                                     (start_time_pattern, prev_epi_start_time),
                                     (framenum_pattern, 'frame_%s' % new_framenum)]
                        for pair in sub_pairs:
                            new_img_file = re.sub(pair[0], pair[1], new_img_file)

                        if not troubleshoot:                        
                            os.rename(img_file, new_img_file)
                        image_files[prev_eIdx].append(new_img_file)
                    
                    image_files[eIdx] = 0
                                            
                    event_frame_nums [prev_eIdx, cIdx] = event_frame_nums[prev_eIdx, cIdx] + len(event_timestamps)
                    event_frame_nums [eIdx, cIdx]      = np.nan
                    event_start_times[eIdx, cIdx]      = np.nan

                    timestamps[prev_eIdx].extend(event_timestamps)
                    timestamps[eIdx] = 0
                    
    return event_frame_nums, event_start_times

def fix_episode_numbers(event_frame_nums, ncams, cam_timestamps, cam_event_image_files, event_start_times, event_start_times_original, date, sNum):
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
    
    for cIdx, (timestamps, image_files) in enumerate(zip(cam_timestamps, cam_event_image_files)):
        for empty_range, epi_range in zip(empty_episode_ranges, episode_ranges_to_edit):
            # for new_epi_idx, start_time, epi_images, old_epi_idx in zip(empty_range, 
            #                                                             event_start_times[epi_range[0] : epi_range[1]+1, cIdx],
            #                                                             image_files[epi_range[0] : epi_range[1]+1],
            #                                                             range(epi_range[0], epi_range[1]+1)):
            #     new_epi_num = str(new_epi_idx + 1).zfill(3)
            for start_time, epi_images, old_epi_idx in zip(event_start_times[epi_range[0] : epi_range[1]+1, cIdx],
                                                           image_files[epi_range[0] : epi_range[1]+1],
                                                           range(epi_range[0], epi_range[1]+1)):
                new_epi_idx = empty_range.start
                new_epi_num = str(new_epi_idx + 1).zfill(3)
                old_epi_num = str(np.where(event_start_times_original[:, cIdx] == start_time)[0][-1] + 1).zfill(3)
                empty_range = range(empty_range.start+1, empty_range.stop+1)
                
                if re.findall(event_pattern, epi_images[0])[0].split('event_')[-1] == old_epi_num:
                    print('\n%s, session %d: Replacing original episode number %s with new episode number %s\n' % (date, 
                                                                                                                   sNum, 
                                                                                                                   old_epi_num,
                                                                                                                   new_epi_num), flush=True)
                    for img_file in epi_images:
                        new_img_file = img_file
                        sub_pairs = [(event_pattern, 'event_%s' % new_epi_num)]
                        for pair in sub_pairs:
                            new_img_file = re.sub(pair[0], pair[1], new_img_file)
                            if not troubleshoot:
                                os.rename(img_file, new_img_file)
                    
                    event_frame_nums [new_epi_idx, cIdx] = event_frame_nums [old_epi_idx, cIdx]
                    event_frame_nums [old_epi_idx, cIdx] = np.nan
                    event_start_times[new_epi_idx, cIdx] = event_start_times[old_epi_idx, cIdx]                    
                    event_start_times[old_epi_idx, cIdx] = np.nan
                    
                else:
                    print('\nALERT!!! Episode number %s in the image filenames does not match the expected episode number %s\n' % (re.findall(event_pattern, epi_images[0])[0].split('event_')[-1], 
                                                                                                                                   old_epi_num))
    return event_frame_nums, event_start_times

def find_and_correct_splits_in_all_episodes(jpg_dir,
                                            vid_dir,
                                            marms,
                                            date, 
                                            exp_name,
                                            session_nums,
                                            fps,
                                            ncams):
    
    # collect all session_event_cam combinations
    vid_frame_nums           = [0]*len(session_nums)
    vid_start_times          = [0]*len(session_nums)
    vid_frame_nums_original  = [0]*len(session_nums)
    vid_start_times_original = [0]*len(session_nums)
    for sIdx, sNum in enumerate(session_nums):
        jpg_path = os.path.join(jpg_dir, '%s/%s/%s/session%d' % (exp_name, marms, date, sNum))
        
        print('started collecting episode info', flush=True)
        event_frame_nums, event_start_times, lastEvent = collect_episode_start_times_and_frame_counts(jpg_path, ncams)

        
        event_start_times_original = event_start_times.copy()
        event_frame_nums_original  = event_frame_nums.copy()

        print('started aligning episodes', flush=True)
        
        event_frame_nums, event_start_times = adjust_episodes_to_align_matches(event_frame_nums, 
                                                                               event_start_times, 
                                                                               lastEvent,
                                                                               ncams,
                                                                               jpg_path)       
                
        # if np.all(event_frame_nums == event_frame_nums_original):
        #     print('breaking out of session %d' % sIdx)
        #     continue
        print('started collecting timestamps and filenames', flush=True)

        cam_timestamps, cam_event_image_files = collect_timestamps_and_filenames(jpg_path, ncams, event_start_times)
        
        print('started stitching episodes together', flush=True)

        event_frame_nums, event_start_times = stitch_episodes_together(cam_timestamps, 
                                                                       cam_event_image_files, 
                                                                       event_frame_nums, 
                                                                       event_start_times,
                                                                       fps)
        print('started fixing episode numbers', flush=True)

        event_frame_nums, event_start_times = fix_episode_numbers(event_frame_nums, 
                                                                  ncams, 
                                                                  cam_timestamps, 
                                                                  cam_event_image_files, 
                                                                  event_start_times, 
                                                                  event_start_times_original,
                                                                  date,
                                                                  sNum)
        
        frame_nums_df  = pd.DataFrame(data = event_frame_nums , columns = ['cam%d' % cam for cam in range(1, ncams+1)])
        start_times_df = pd.DataFrame(data = event_start_times, columns = ['cam%d' % cam for cam in range(1, ncams+1)])
        vid_frame_nums [sIdx] = frame_nums_df
        vid_start_times[sIdx] = start_times_df
    
        frame_nums_df_original  = pd.DataFrame(data = event_frame_nums_original , columns = ['cam%d' % cam for cam in range(1, ncams+1)])
        start_times_df_original = pd.DataFrame(data = event_start_times_original, columns = ['cam%d' % cam for cam in range(1, ncams+1)])
        vid_frame_nums_original [sIdx] = frame_nums_df_original
        vid_start_times_original[sIdx] = start_times_df_original

    episode_adjustment_records = {'original_episode_frame_counts'  : vid_frame_nums_original,
                                  'original_episode_start_times'   : vid_start_times_original,
                                  'corrected_episode_frame_counts' : vid_frame_nums,
                                  'corrected_episode_start_times'  : vid_start_times} 
    
    with open(os.path.join(episode_correction_record_path, '%s_%s_%s_episode_correction_record.pkl' % (marms, date, exp_name)), 'wb') as f:
        dill.dump(episode_adjustment_records, f, recurse=True) 

if __name__ == '__main__':
    
    troubleshoot = False
    
    if not troubleshoot:
    
        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-j", "--jpg_dir", required=True, type=str,
            help="path to temporary directory holding jpg files for task and marmoset pair. E.g. /scratch/midway3/daltonm/kinematics_jpgs/")
        ap.add_argument("-v", "--vid_dir", required=True, type=str,
            help="path to directory for task and marmoset pair. E.g. /project/nicho/data/marmosets/kinematics_videos/")
        ap.add_argument("-m", "--marms", required=True, type=str,
         	help="marmoset 4-digit code, e.g. 'JLTY'")
        ap.add_argument("-d", "--date", required=True, type=str,
            help="date of recording in format YYYY_MM_DD")        
        ap.add_argument("-e", "--exp_name", required=True, type=str,
         	help="experiment name, e.g. free, foraging, BeTL, crickets, moths, etc")
        ap.add_argument("-s", "--session_nums", nargs='+', required=True, type=int,
         	help="session numbers (can have multiple entries separated by spaces)")
        ap.add_argument("-f", "--fps", required=True, type=int,
         	help="camera frame rate")
        ap.add_argument("-n", "--ncams", required=True, type=int,
         	help="number of cameras")
        args = vars(ap.parse_args())
                
        session_nums = [int(num) for num in args['session_nums']]
    
        try:
            task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        except:
            task_id = 0
    else:
        session_nums = [1]
        args = {'date':'2025_01_25',
                'vid_dir':'/project/nicho/data/marmosets/kinematics_videos',
                'exp_name': 'baseline',
                'marms': 'TYTR',
                'fps':150,
                'ncams':5,
                'jpg_dir':'/scratch/midway3/snjohnso/kinematics_jpgs'}
        task_id = 0
        
    data_path = os.path.join(args['vid_dir'], args['exp_name'], args['marms'], args['date'])
    episode_correction_record_path = os.path.join(data_path, 'metadata_from_kinematics_processing')
    videos_path                    = os.path.join(data_path, 'avi_videos')
    os.makedirs(episode_correction_record_path, exist_ok=True)
    
    if task_id == 0:
        
        print(f'\n\n Beginning check_for_episode_splits code at {time.strftime("%c", time.localtime())}\n\n', flush=True)
        
        find_and_correct_splits_in_all_episodes(args['jpg_dir'],
                                                args['vid_dir'],
                                                args['marms'], 
                                                args['date'], 
                                                args['exp_name'], 
                                                session_nums, 
                                                args['fps'], 
                                                args['ncams'])
        
        print(f'\n\n Completed check_for_episode_splits code at {time.strftime("%c", time.localtime())}\n\n', flush=True)

    else:
        
        print(f'\n\n Waiting for task_0 to complete check/correction of episode splits and misalignment. Will move on when \n a directory is created at {videos_path} by task_0. Current time is {time.strftime("%c", time.localtime())}\n\n', flush=True)
        
        jpg2avi_started = False
        while not jpg2avi_started:
            jpg2avi_started = os.path.isdir(videos_path)
            
        print(f'\n\n Moving on to jpg2avi conversion at {time.strftime("%c", time.localtime())}\n\n', flush=True)

 