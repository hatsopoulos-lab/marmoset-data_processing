# -*- coding: utf-8 -*-
"""
Created on June 07 2022

@author: Dalton
"""

# An automated processing script for converting jpg files into videos.
# An automated processing script for converting jpg files into videos.
                                                                     
# Example: sudo Documents/camera_control_code/jpg2avi_apparatus.sh 'TYJL' '2021_01_08' 'foraging' '1' '150'

#                                                                marmCode    date      expName session  framerate

import glob
import re
import os
import subprocess
import argparse
import time
import numpy as np

class paths: 
    processing_code   = '/project/nicho/projects/marmosets/code_database/data_processing/kinematics'    

def convert_jpg_to_video(jpg_dir, vid_dir, marms, dates, exp, session_nums, fps, ncams, transpose, video_ext = 'avi'):

    for date in dates:
        video_path = os.path.join(vid_dir, '%s/%s/%s/unfiltered_videos' % (exp, marms, date))
        drop_record_path = os.path.join(vid_dir, '%s/%s/%s/drop_records' % (exp, marms, date))
        calibration_path = os.path.join(vid_dir, '%s/%s/%s/calibration' % (exp, marms, date))
        os.makedirs(video_path, exist_ok=True)
        os.makedirs(drop_record_path, exist_ok=True)
        os.makedirs(calibration_path, exist_ok=True)
        
        prev_sum_video_sizes = 0
        updated_sum = 10
        while updated_sum > prev_sum_video_sizes or any(np.array([os.path.getsize('%s/%s' % (video_path, f)) for f in os.listdir('%s/.' % video_path)]) < 10000): 
            prev_sum_video_sizes = sum(os.path.getsize('%s/%s' % (video_path, f)) for f in os.listdir('%s/.' % video_path))
            for sNum in session_nums:
                jpg_path = os.path.join(jpg_dir, '%s/%s/%s/session%d' % (exp, marms, date, sNum))

                for cam, trans in zip(range(1, ncams+1), transpose):
                    event_pattern      = re.compile('event_\d{3}')
                    start_time_pattern = re.compile('\d{4}-\d{2}.jpg')

                    jpg_file = sorted(glob.glob(os.path.join(jpg_path, 'jpg_cam%d' % cam, '*')))[-1]
                    lastEvent = int(re.findall(event_pattern, jpg_file)[0].split('event_')[-1])
                    print('creating avi videos for cam' + str(cam))
                
                    # If you want to change filename conventions or video params, this is where to do so (in the ffmpeg line). 
                    # -s flag adjusts resolution. 
                    # -pattern_type glob grabs all filenames that start as written and end in .jpg, and turns these into a video. 
                    # -crf flag changes the compression ratio, with higher numbers = more compression. 
                    # -vcodec changes the video codec used, i wouldn't touch this unless you get an error. 
                    # The last argument is the filepath for the video to be stored.     
                    for eNum in range(1, lastEvent+1):
                        event=str(eNum).zfill(3)
                        cam_img_path = os.path.join(jpg_path, 'jpg_cam%d' % cam)
                        tmp_event_image_file = glob.glob(os.path.join(cam_img_path, '*cam%d_event_%s*' % (cam, event)))[0]
                        start_time = re.findall(start_time_pattern, tmp_event_image_file)[0].split('.jpg')[0]
                        outvidfile = os.path.join(video_path, '%s_%s_%s_%s_s%d_e%s_cam%d.%s' % (marms, date, start_time, exp, sNum, event, cam, video_ext))
                        if os.path.exists(outvidfile):
                            print('%s already exists - skipping' % outvidfile, flush=True)
                            time.sleep(2)
                            updated_sum = sum(os.path.getsize('%s/%s' % (video_path, f)) for f in os.listdir('%s/.' % video_path))
                            continue
                        else:
                            print('%s DOES NOT EXIST' % outvidfile, flush=True)

                        subprocess.call(['python',  
                                         os.path.join(paths.processing_code, 'fix_drops_during_vid_conversion.py'), 
                                         os.path.join(cam_img_path, '%s_%s_%s_session_%d_cam%d_event_%s_frame_*' % (marms, date, exp, sNum, cam, event)), 
                                         drop_record_path,
                                         str(fps)])
                    
                        if int(trans) == -1: 
                            subprocess.call(['ffmpeg', 
                                             '-r', str(fps), 
                                             '-f', 'image2', 
                                             '-s', '1440x1080', 
                                             '-pattern_type', 'glob',
                                             '-i', os.path.join(cam_img_path, '%s_%s_%s_session_%d_cam%d_event_%s_frame_*.jpg' % (marms, date, exp, sNum, cam, event)), 
                                             '-crf', '15',
                                             '-vcodec', 'libx264', 
                                             outvidfile])
                        else:
                            subprocess.call(['ffmpeg', 
                                             '-r', str(fps), 
                                             '-f', 'image2', 
                                             '-s', '1440x1080', 
                                             '-pattern_type', 'glob',
                                             '-i', os.path.join(cam_img_path, '%s_%s_%s_session_%d_cam%d_event_%s_frame_*.jpg' % (marms, date, exp, sNum, cam, event)), 
                                             '-vf', 'transpose=%s' % trans,
                                             '-crf', '15',
                                             '-vcodec', 'libx264', 
                                             outvidfile])
                        
                        updated_sum = sum(os.path.getsize('%s/%s' % (video_path, f)) for f in os.listdir('%s/.' % video_path))

        calib_image_path = glob.glob(os.path.join(jpg_dir, '%s/%s/%s/*calib*' % (exp, marms, date)))[0]
        prev_sum_video_sizes = 0
        updated_sum = 10
        while updated_sum > prev_sum_video_sizes or any(np.array([os.path.getsize('%s/%s' % (calibration_path, f)) for f in os.listdir('%s/.' % calibration_path)]) < 10000): 
            prev_sum_video_sizes = sum(os.path.getsize('%s/%s' % (calibration_path, f)) for f in os.listdir('%s/.' % calibration_path))
            for cam, trans in zip(range(1, ncams+1), transpose):
                calib_cam_path = os.path.join(calib_image_path, 'cam%d' % cam)
                updated_sum = sum(os.path.getsize('%s/%s' % (calibration_path, f)) for f in os.listdir('%s/.' % calibration_path))

            

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--jpg_dir", required=True,
        help="path to temporary directory holding jpg files for task and marmoset pair. E.g. /scratch/midway3/daltonm/kinematics_jpgs/")
    ap.add_argument("-v", "--vid_dir", required=True,
        help="path to directory for task and marmoset pair. E.g. /project/nicho/data/marmosets/kinematics_videos/")
    ap.add_argument("-m", "--marms", required=True,
     	help="marmoset 4-digit code, e.g. 'JLTY'")
    ap.add_argument("-d", "--dates", nargs='+', required=True,
     	help="date(s) of recording (can have multiple entries separated by spaces)")
    ap.add_argument("-e", "--exp_name", required=True,
     	help="experiment name, e.g. free, foraging, BeTL, crickets, moths, etc")
    ap.add_argument("-s", "--session_nums", nargs='+', required=True,
     	help="session numbers (can have multiple entries separated by spaces)")
    ap.add_argument("-f", "--fps", required=True,
     	help="camera frame rate")
    ap.add_argument("-n", "--ncams", required=True,
     	help="number of cameras")
    ap.add_argument("-t", "--video_transpose", nargs='+', required=True,
     	help="video transpose codes (see ffmpeg docs for details). Use '-1' for no transpose. E.g. '2 2 1 1' ")
    args = vars(ap.parse_args())
    
    session_nums = [int(num) for num in args['session_nums']]

    jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    time.sleep(jobid)

    convert_jpg_to_video(args['jpg_dir'],
                         args['vid_dir'],
                         args['marms'], 
                         args['dates'], 
                         args['exp_name'], 
                         session_nums, 
                         args['fps'], 
                         int(args['ncams']), 
                         transpose = args['video_transpose'],
                         video_ext = 'avi')
