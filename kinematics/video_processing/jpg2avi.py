# -*- coding: utf-8 -*-
"""
Created on June 07 2022
Edited on November 11 2023

@author: Dalton
"""

# An automated processing script for converting jpg files into videos.

import glob
import re
import os
import cv2
import subprocess
import argparse
import time
import numpy as np
import pandas as pd
import shutil
import itertools
from pathlib import Path

event_pattern = re.compile('event_\d{3,5}_')
event_pattern_backup = re.compile('_e_\d{3,5}_')

class paths: 
    processing_code   = '/project/nicho/projects/marmosets/code_database/data_processing/kinematics/video_processing'    

def filterFrame(frame, clahe):
    if frame is not None:
        yframe = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yframe[:,:,0] = clahe.apply(yframe[:,:,0])
        frame = cv2.cvtColor(yframe, cv2.COLOR_YUV2BGR)
    return frame

def apply_clahe_filter_to_all_images(jpg_files): 

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    for img_path in jpg_files:
        # start = time.time()
        frame = cv2.imread(img_path)
        frame = filterFrame(frame, clahe)
        # print(f'{img_path}')
        # print(f'{np.shape(frame)}', flush=True)
        cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
        # print(time.time() - start)
    return

def fix_dropped_frames(jpg_files, jpgPattern, drop_record_path, fps):

    period_ns = 1/fps * 1e9

    first_frame = Path(jpg_files[0])
    try:
        event = re.findall(event_pattern, first_frame.stem)[0].split('event_')[1][:-1] 
    except:
        event = re.findall(event_pattern_backup, first_frame.stem)[0].split('e_')[1][:-1] 
        
    try:
        subject_date_exp = first_frame.stem.split('_session')[0]
    except:
        subject_date_exp = first_frame.stem.split('_s')[0]

    timestamps = []
    frameNums = []
    frameIdx = []
    for fr, file in enumerate(jpg_files):
        timestamps.append(int(file.split('currentTime_')[1][:-12]))
        frameNums.append(int(file.split('frame_')[1][:7]))
        frameIdx.append(fr)

    frameDiffs = np.diff(frameNums)
    dropFrames = np.where(frameDiffs > 1)[0]


    dropRecord = []
    
    # add dummy frames at beginning if necessary (very rare, seen only once on 2023_08_09, moth_s1_e1_cam1 and moth_free_s1_e1_cam2)
    if frameNums[0] > 1:
        dropFrames = np.insert(dropFrames, 0, np.array(range(1, frameNums[0])))
        
    # check if last frame was dropped by comparing to other camera folders
    bases = [jpgPattern.split('jpg_cam')[0]]

    lastFrameNums = []
    for base in bases:
        cam_folders = glob.glob(os.path.join(base, 'jpg_cam*'))
        for f in cam_folders:
            lastFrameNums.append(int(sorted(glob.glob(os.path.join(f, '%s*event_%s*' %(subject_date_exp, event))))[-1].split('frame_')[1][:7]))            

    # append values to frameNums and dropFrames so that the dropped frame(s) at the
    # end will be replaced
    if frameNums[-1] < max(lastFrameNums):
        frameDiffs = np.append(frameDiffs, max(lastFrameNums) - frameNums[-1] + 1)
        dropFrames = np.append(dropFrames, len(frameNums)-1)
        dropRecord.append(max(lastFrameNums))

    if len(dropFrames) > 0:
        sessDir, cam = os.path.split(os.path.dirname(jpg_files[0]))
        cam = cam.split('jpg_')[1]
        dateDir, session = os.path.split(sessDir)  
        
        drop_record_file = os.path.join(drop_record_path, f'{subject_date_exp}_{session}_event_{event}_{cam}_droppedFrames.txt')
        print(f'Identified some missing frames. Replacing with copies of previous good frame. Record of copies stored at \n{drop_record_file}')
        
        with open(drop_record_file, 'w') as f:
        
            firstGoodFrame, firstGoodTime, firstGoodFile = frameNums[0], timestamps[0], jpg_files[0]        
            for dFr in dropFrames:
                
                if dFr < frameNums[0]:
                    newFrame = dFr
                    newFile  = firstGoodFile.replace(f'frame_{str(firstGoodFrame).zfill(7)}', f'frame_{str(newFrame).zfill(7)}')
                    newFile  = newFile.replace(f'currentTime_{str(firstGoodTime)}', f'currentTime_{str(int(firstGoodTime - period_ns*(firstGoodFrame-dFr)))}')
                    shutil.copyfile(firstGoodFile, newFile)
                    dropRecord.append(newFrame)
                    
                else:
                    lastGoodFrame, lastGoodTime, lastGoodFile = frameNums[dFr], timestamps[dFr], jpg_files[dFr]        
                    for copyNum in range(1, frameDiffs[dFr]):
                        newFrame = lastGoodFrame + copyNum
                        newFile = lastGoodFile.replace(f'frame_{str(lastGoodFrame).zfill(7)}', f'frame_{str(newFrame).zfill(7)}')
                        newFile = newFile.replace(f'currentTime_{str(lastGoodTime)}', f'currentTime_{str(int(lastGoodTime + period_ns*copyNum))}')
        
                        shutil.copyfile(lastGoodFile, newFile)
                        
                        dropRecord.append(newFrame)
            
            dropRecord = sorted(dropRecord)
    
            if len(dropRecord) > 1 and dropRecord[-1] == dropRecord[-2]:
                dropRecord = dropRecord[:-1]
    
            for fr in dropRecord:
                f.write(str(fr) + ',') 
    
    return


        
def create_calib_videos(jpg_dir, exp, marms, date, calibration_path, calib_name, video_ext):
        
    calib_image_path = glob.glob(os.path.join(jpg_dir, exp, marms, date, calib_name))[0]
    for cam, trans in zip(cams, transpose):
        
        calib_cam_path = os.path.join(calib_image_path, 'cam%d' % cam)
        
        outvid = os.path.join(calibration_path, f'{marms}_{date}_{exp}_{calib_name}_cam{cam}.{video_ext}')
        if os.path.exists(outvid):
            print(f'{outvid.split(calibration_path)[-1]} already exists - skipping', flush=True)
            continue
        
        print(f'Creating {outvid.split(calibration_path)[-1]}', flush=True)
        
        if int(trans) == -1: 
            subprocess.call(['ffmpeg', 
                             '-r', '20', 
                             '-f', 'image2', 
                             '-s', '1440x1080', 
                             '-pattern_type', 'glob',
                             '-i', os.path.join(calib_cam_path, '*frame_*.jpg'), 
                             '-crf', '15',
                             '-vcodec', 'libx264', 
                             outvid])
        else:
            subprocess.call(['ffmpeg', 
                             '-r', '20', 
                             '-f', 'image2', 
                             '-s', '1440x1080', 
                             '-pattern_type', 'glob',
                             '-i', os.path.join(calib_cam_path, '*frame_*.jpg'),  
                             '-vf', 'transpose=%s' % trans,
                             '-crf', '15',
                             '-vcodec', 'libx264', 
                             outvid])

    return

def collect_jpg_info(jpg_dir, vid_dir, exp, marms, date, session_nums, video_ext):

    # collect all date/session/event/cam filepaths and image counts
    video_conv_info = {'date'       : [],
                       'session'    : [],
                       'event'      : [],
                       'cam'        : [],
                       'imageCount' : [],
                       'jpgPatterns': [],
                       'outVidPaths': [],
                       'transpose'  : []}    
        
    video_task_completion_path = os.path.join(vid_dir, exp, marms, date, 'video_task_completion_tmp_files')    
    clahe_task_completion_path = os.path.join(vid_dir, exp, marms, date, 'clahe_task_completion_tmp_files')    
    os.makedirs(video_task_completion_path, exist_ok=True)
    os.makedirs(clahe_task_completion_path, exist_ok=True)
    
    video_path           = os.path.join(vid_dir, exp, marms, date, 'avi_videos')
    drop_record_path     = os.path.join(vid_dir, exp, marms, date, 'drop_records' )
    calibration_path     = os.path.join(vid_dir, exp, marms, date, 'calibration')
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(drop_record_path, exist_ok=True)
    os.makedirs(calibration_path, exist_ok=True)

    # collect all session_event_cam combinations and assign idx cutoffs to each slurm task
    for sNum in session_nums:
        jpg_path = os.path.join(jpg_dir, exp, marms, date, f'session{sNum}')

        for cam, trans in zip(cams, transpose):
            event_pattern      = re.compile('event_\d{3}')
            start_time_pattern = re.compile('\d{4}-\d{2}.jpg')

            print(os.path.join(f'\n{jpg_path}', f'jpg_cam{cam}', '*'), flush=True)

            jpg_files = sorted(glob.glob(os.path.join(jpg_path, 'jpg_cam%d' % cam, '*')))    
            print(f'got {len(jpg_files)} jpg files', flush=True)
            lastEvent = int(re.findall(event_pattern, jpg_files[-1])[0].split('event_')[-1])
            print(f'last event is {lastEvent}\n', flush=True)               
            # If you want to change filename conventions or video params, this is where to do so (in the ffmpeg line). 
            # -s flag adjusts resolution. 
            # -pattern_type glob grabs all filenames that start as written and end in .jpg, and turns these into a video. 
            # -crf flag changes the compression ratio, with higher numbers = more compression. 
            # -vcodec changes the video codec used, i wouldn't touch this unless you get an error. 
            # The last argument is the filepath for the video to be stored.     
            for eNum in range(1, lastEvent+1):
                print(f'eventNum = {eNum}', flush=True)
                event=str(eNum).zfill(3)
                cam_img_path = os.path.join(jpg_path, f'jpg_cam{cam}')
                tmp_event_image_file = glob.glob(os.path.join(cam_img_path, f'*cam{cam}_event_{event}*'))[0]
                start_time = re.findall(start_time_pattern, tmp_event_image_file)[0].split('.jpg')[0]
                outvid = os.path.join(video_path,  f'{marms}_{date}_{start_time}_{exp}_s{sNum}_e{event}_cam{cam}.{video_ext}')
                jpgPattern = os.path.join(cam_img_path, f'{marms}_{date}_*_session_{sNum}_cam{cam}_event_{event}_frame_*')
                image_count = len(glob.glob(jpgPattern))
                
                video_conv_info['jpgPatterns'].append(jpgPattern)
                video_conv_info['outVidPaths'].append(outvid)
                video_conv_info['transpose'].append(trans)
                video_conv_info['imageCount'].append(image_count)
                video_conv_info['event'].append(eNum)
                video_conv_info['date'].append(date)
                video_conv_info['session'].append(sNum)
                video_conv_info['cam'].append(cam)
    
    event_info = pd.DataFrame.from_dict(data=video_conv_info)
    event_info.sort_values(by=['imageCount', 'cam'], inplace=True, ignore_index=True, ascending=False)
    event_info['task_id_video'] = np.full((event_info.shape[0],), -1, dtype=int)  
    
    paths_dict = {'video_completion'   : video_task_completion_path,
                  'clahe_completion'   : clahe_task_completion_path,
                  'drop_records'       : drop_record_path, 
                  'calibration_videos' : calibration_path}
    
    return event_info, paths_dict

def assign_events_and_images_to_tasks(event_info, task_id):
    # assign events to tasks
    if n_tasks <= event_info.shape[0]:
        # assign video production to tasks
        for idx, tmp_info in event_info.iterrows():
            if idx < n_tasks:
                event_info.at[idx, 'task_id_video'] = idx
            else:
                task_image_totals = event_info.loc[:, ['imageCount', 'task_id_video']].groupby('task_id_video').sum()
                task_image_totals = task_image_totals[task_image_totals.index != -1]
                event_info.at[idx, 'task_id_video'] = int(task_image_totals['imageCount'].argmin())
        # assign clahe filtering to the same task
        event_info['task_id_clahe'] = event_info['task_id_video']
        # clahe_task_info = event_info.loc[event_info['task_id_clahe'] == task_id]
        # clahe_task_info['task_id_clahe']
    else:
        # assign 1 video to each task until all videos are assigned
        for idx, tmp_info in event_info.iterrows():
            event_info.at[idx, 'task_id_video'] = idx
        # assign chunks of jpegs to tasks for clahe filtering (this is to break up the free behavior videos that are very long)
        event_info['task_id_clahe'] = [[] for idx in range(event_info.shape[0])] 
        extra_tasks_per_video = int(np.floor(n_tasks / event_info.shape[0])) - 1
        first_empty_task = event_info['task_id_video'].max() + 1 
        for idx, tmp_info in event_info.iterrows():
            event_info.at[idx, 'task_id_clahe'] = [idx] + list(range(first_empty_task, first_empty_task+extra_tasks_per_video))
            first_empty_task = event_info.at[idx, 'task_id_clahe'][-1] + 1
            
    clahe_task_info = event_info.copy()  
    video_task_info = event_info.loc[event_info['task_id_video'] == task_id]
    
    return video_task_info, clahe_task_info

def run_clahe_filtering(clahe_task_info, vid_dir, task_id, clahe_task_completion_path):
    
    print(clahe_task_info['task_id_clahe'])
    
    if type(clahe_task_info['task_id_clahe'].iloc[0]) != list:
        clahe_completed = True
        all_clahe_tasks = sorted(list(clahe_task_info['task_id_clahe']))   
    else:
        clahe_completed = False
        all_clahe_tasks = sorted(list(itertools.chain.from_iterable(clahe_task_info['task_id_clahe'])))   

    for idx, clahe_info in clahe_task_info.iterrows():
        
        clahe_tasks = clahe_info['task_id_clahe']
        if type(clahe_tasks) != list:
            clahe_tasks = [clahe_tasks]
            
        if task_id not in clahe_tasks:
            continue
        
        outvid = clahe_info['outVidPaths']
        if os.path.exists(outvid):
            print(f'{outvid.split(vid_dir)[-1]} already exists - skipping CLAHE filter step on jpgs', flush=True)
            with open(os.path.join(clahe_task_completion_path, f'clahe_completed_task_{task_id}.txt'), 'w') as fp: 
                pass   
            continue
        
        else:
            task_idx = [i for i, task in enumerate(clahe_tasks) if task == task_id][0]
            jpgPattern = clahe_info['jpgPatterns']            
            all_jpg_files = sorted(glob.glob(jpgPattern))
            jpg_idx_bounds = [int(np.floor(bound)) for bound in np.linspace(0, len(all_jpg_files), len(clahe_tasks)+1)]
            task_jpg_files = all_jpg_files[jpg_idx_bounds[task_idx] : jpg_idx_bounds[task_idx+1]]
    
            print(f'\nWorking on images located at \n{jpgPattern}\n', flush=True)
            print(f'Working on frame {task_jpg_files[0].split("frame_")[-1].split("_")[0]} to frame {task_jpg_files[-1].split("frame_")[-1].split("_")[0]}', flush=True)
            print(f'\nApplying CLAHE filter, starting {time.strftime("%c", time.localtime())}\n', flush=True)
            apply_clahe_filter_to_all_images(task_jpg_files)
            print(f'Finished CLAHE filter, {time.strftime("%c", time.localtime())}\n', flush=True)

            with open(os.path.join(clahe_task_completion_path, f'clahe_completed_task_{task_id}.txt'), 'w') as fp: 
                pass  
    
    first_check = True
    while not clahe_completed:
        if not first_check:
            time.sleep(60*5)
        first_check=False
        clahe_completion_files = glob.glob(os.path.join(clahe_task_completion_path, 'clahe_completed*.txt'))
        clahe_completed_tasks = sorted([int(os.path.basename(f).split('task_')[-1].split('.txt')[0]) for f in clahe_completion_files])
        clahe_completed = True if clahe_completed_tasks == all_clahe_tasks else False
        print(f'Clahe {"is" if clahe_completed else "NOT"} completed. \n Waiting 5 minutes.', flush=True)
        
    return

def run_video_production(video_task_info, vid_dir, drop_record_path, fps, video_task_completion_path, task_id):
    if video_task_info.size == 0: 
        print('This task is not responsible for any video production.', flush=True)
        return
    
    print(f'\nCreating avi videos listed below in "{vid_dir}":\n')
    for outvid in video_task_info['outVidPaths']:
        print(outvid.split(vid_dir)[-1])
    print('', flush=True)

    for idx, vid_info in video_task_info.iterrows():
        # If you want to change filename conventions or video params, this is where to do so (in the ffmpeg line). 
        # -s flag adjusts resolution. 
        # -pattern_type glob grabs all filenames that start as written and end in .jpg, and turns these into a video. 
        # -crf flag changes the compression ratio, with higher numbers = more compression. 
        # -vcodec changes the video codec used, i wouldn't touch this unless you get an error. 
        # The last argument is the filepath for the video to be stored.     
        
        outvid     = vid_info['outVidPaths']
        jpgPattern = vid_info['jpgPatterns']
        trans      = vid_info['transpose'  ] 
        
        if os.path.exists(outvid):
            print(f'{outvid.split(vid_dir)[-1]} already exists - skipping video production', flush=True)
            continue

        jpg_files = sorted(glob.glob(jpgPattern))

        print(f'Looking for and fixing dropped frames, starting {time.strftime("%c", time.localtime())}', flush=True)
        print(f'Found {len(jpg_files)} frames', flush=True)
        fix_dropped_frames(jpg_files, jpgPattern, drop_record_path, fps)
        print(f'Done addressing dropped frames, {time.strftime("%c", time.localtime())}', flush=True)
    
        print(f'Creating {outvid.split(vid_dir)[-1]}, starting {time.strftime("%c", time.localtime())}', flush=True)
       
        if int(trans) == -1: 
            subprocess.call(['ffmpeg', 
                             '-r', str(fps), 
                             '-f', 'image2', 
                             '-s', '1440x1080', 
                             '-pattern_type', 'glob',
                             '-i', '%s.jpg' % jpgPattern, 
                             '-crf', '15',
                             '-vcodec', 'libx264', 
                             outvid])
        else:
            subprocess.call(['ffmpeg', 
                             '-r', str(fps), 
                             '-f', 'image2', 
                             '-s', '1440x1080', 
                             '-pattern_type', 'glob',
                             '-i', '%s.jpg' % jpgPattern, 
                             '-vf', 'transpose=%s' % trans,
                             '-crf', '15',
                             '-vcodec', 'libx264', 
                             outvid])
        print(f'Done with {outvid.split(vid_dir)[-1]}, {time.strftime("%c", time.localtime())}', flush=True)
    
    with open(os.path.join(video_task_completion_path, f'behavior_videos_completed_task_{task_id}.txt'), 'w') as fp: 
        pass
    
    return

def run_calibration_video_production(video_task_completion_path, jpg_dir, exp, 
                                     marms, date, calibration_path, calib_name, 
                                     video_ext, task_id, last_task, num_video_task_files):
    calib_completed = False
    behavior_completed = False
    first_check = True
    if calib_name is None:
        calib_completed = True
    else:
        calib_completion_file = os.path.join(video_task_completion_path, 'calib_videos_completed.txt')
        if task_id == 0:
            
            create_calib_videos(jpg_dir, exp, marms, date, calibration_path, calib_name, video_ext)
            
            with open(calib_completion_file, 'w') as fp: 
                pass
        
        # hold the last task here until all videos are created, then move along to processing signals (in another python code)
        if task_id == last_task:
            while not (behavior_completed and calib_completed):
                if not first_check:
                    time.sleep(60*5)
                first_check = False
                behavior_video_completion_files = glob.glob(os.path.join(video_task_completion_path, 'behavior_videos_completed*.txt'))
                print(f'n_task_files = {num_video_task_files}, completed_tasks = {len(behavior_video_completion_files)}', flush=True)
                behavior_completed = True if len(behavior_video_completion_files) == num_video_task_files else False
                calib_completed    = os.path.isfile(calib_completion_file)
                print(f'Behavior videos completed: {behavior_completed}, Calibration videos completed: {calib_completed}.\n Waiting 5 minutes.', flush=True)

    
def convert_jpg_to_video(jpg_dir, vid_dir, marms, date, exp, session_nums, fps, 
                         cams, transpose, calib_name = None, apply_clahe = True, 
                         video_ext = 'avi', task_id = 0, n_tasks = 1, last_task=0):

    event_info, paths_dict = collect_jpg_info(jpg_dir, vid_dir, exp, marms, date, session_nums, video_ext)
    
    clahe_task_completion_path = paths_dict['clahe_completion'] 
    video_task_completion_path = paths_dict['video_completion']
    drop_record_path           = paths_dict['drop_records']
    calibration_path           = paths_dict['calibration_videos']
    
    video_task_info, clahe_task_info = assign_events_and_images_to_tasks(event_info, task_id)

    if apply_clahe:
        run_clahe_filtering(clahe_task_info, vid_dir, task_id, clahe_task_completion_path)
    else: 
        print('\napply_clahe = False. Skipping clahe filtering\n')

    run_video_production(video_task_info, vid_dir, drop_record_path, fps, video_task_completion_path, task_id)
    
    num_video_task_files = clahe_task_info.shape[0] if clahe_task_info.shape[0] <= n_tasks else n_tasks 
    
    run_calibration_video_production(video_task_completion_path, jpg_dir, exp, 
                                     marms, date, calibration_path, calib_name, 
                                     video_ext, task_id, last_task, num_video_task_files)                    
    
    print(f'\n Finished jpg2avi code on {time.strftime("%c", time.localtime())}\n', flush=True)

    if task_id == last_task:
        shutil.rmtree(clahe_task_completion_path, ignore_errors=True)
        shutil.rmtree(video_task_completion_path, ignore_errors=True)
        
    return

def convert_string_inputs_to_int_float_or_bool(orig_var):
    if type(orig_var) == str:
        orig_var = [orig_var]
    
    converted_var = []
    for v in orig_var:
        v = v.lower()
        try:
            v = int(v)
        except:
            pass
        try:
            v = float(v)
        except:
            v = None  if v == 'none'  else v
            v = True  if v == 'true'  else v
            v = False if v == 'false' else v 
        converted_var.append(v)
    
    if len(converted_var) == 1:
        converted_var = converted_var[0]
            
    return converted_var

if __name__ == '__main__':
    # construct the argument parse and parse the arguments

    debugging = False

    if not debugging:
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
        ap.add_argument("-cm", "--cams", nargs='+', required=True, type=int,
         	help="number of cameras")
        ap.add_argument("-t", "--video_transpose", nargs='+', required=True, type=int,
         	help="video transpose codes (see ffmpeg docs for details). Use '-1' for no transpose. E.g. '2 2 1 1' ")
        ap.add_argument("-c", "--calib_name", required=True, type=str,
         	help="name of calibration folder (e.g. 'calib', 'pre_calib', 'None' ")
        ap.add_argument("-cl", "--apply_clahe", required=True, type=str,
         	help="name of calibration folder (e.g. 'calib', 'pre_calib', 'None' ")
        args = vars(ap.parse_args())
    else:
        args = {'jpg_dir'        : '/scratch/midway3/snjohnso/kinematics_jpgs',
                'vid_dir'        : '/project/nicho/data/marmosets/kinematics_videos',
                'marms'          : 'TYTR',
                'date'           : '2025_01_25',
                'exp_name'       : 'baselineFree',
                'session_nums'   : [1],
                'fps'            : 30,
                'cams'           : [ 1,  2,  3,  4],
                'video_transpose': [-1, -1, -1, -1],
                'calib_name'     : 'calib',
                'apply_clahe'    : 'True'}
    try:
        task_id   = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        n_tasks   = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
        last_task = int(os.getenv('SLURM_ARRAY_TASK_MAX'))
    except:
        task_id = 0
        n_tasks = 20
        last_task = task_id
    print(f'task_id = {task_id}', flush=True)
    
    now = time.localtime()
    print(f'\n Beginning jpg2avi code on {time.strftime("%c", now)}\n', flush=True)
    
    # session_nums = [int(num) for num in args['session_nums']]
    calib_name = args['calib_name']
    if calib_name == 'None':
        calib_name = None

    if len(args['cams']) != len(args['video_transpose']):
        print(args['cams'])
        print(f'length cams = {len(args["cams"])} and length vid_transpose = {len(args["video_transpose"])}', flush=True)
        raise SystemExit('the length of the "cams" and "vid_transpose" inputs must be the same')
    else:
        transpose = []
        cams = []
        for cam, trans in zip(args['cams'], args['video_transpose']):
            if cam != -1:
                trans = trans if trans in [0, 1, 2, 3] else -1
                transpose.append(trans)
                cams.append(cam)   

    apply_clahe = convert_string_inputs_to_int_float_or_bool(args['apply_clahe'])                 
    
    convert_jpg_to_video(args['jpg_dir'],
                         args['vid_dir'],
                         args['marms'], 
                         args['date'], 
                         args['exp_name'], 
                         args['session_nums'], 
                         args['fps'], 
                         cams = cams, 
                         transpose = transpose,
                         calib_name = calib_name,
                         apply_clahe = apply_clahe,
                         video_ext = 'avi',
                         task_id = task_id,
                         n_tasks = n_tasks,
                         last_task=last_task)
