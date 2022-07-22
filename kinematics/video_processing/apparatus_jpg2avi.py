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

class paths:
    kinematics_videos = '/project/nicho/data/marmosets/kinematics_videos' 
    processing_code   = '/project/nicho/projects/marmosets/code_database/processing'    

def convert_jpg_to_video(marms, dates, exp, session_nums, fps, ncams, video_ext = 'avi'):

    for date in dates:
        video_path = os.path.join(paths.kinematics_videos, '%s/%s/%s/unfiltered_videos' % (exp, marms, date))
        os.makedirs(video_path, exist_ok=True)
        for sNum in session_nums:
            jpg_path = os.path.join(paths.kinematics_videos, '%s/%s/%s/session%d' % (exp, marms, date, sNum))

            event_pattern      = re.compile('event_\d{3}')
            start_time_pattern = re.compile('\d{4}-\d{2}.jpg')
                
            # This chunck pulls out the last event number so a for loop can run to convert videos in order with conistent naming conventions.
            jpg_file = sorted(glob.glob(os.path.join(jpg_path, 'jpg_cam1', '*')))[-1]
            lastEvent = int(re.findall(event_pattern, jpg_file)[0].split('event_')[-1])
            
            for cam in range(1, ncams+1):
            
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
                    tmp_event_image_file = glob.glob(cam_img_path, '*cam%d_event_%s*' % (cam, event))[0]
                    start_time = re.findall(start_time_pattern, tmp_event_image_file)[0].split('.jpg')[0]
                    
                    subprocess.call(['python',  
                                     os.path.join(paths.processing_code, 'fix_drops_during_vid_conversion.py'), 
                                     os.path.join(cam_img_path, '%s_%s_%s_session_%d_cam%d_event_%s_frame_*' % (marms, date, exp, sNum, cam, event)), 
                                     str(fps)])
                    
                    subprocess.call(['ffmpeg', 
                                     '-r', str(fps), 
                                     '-f', 'image2', 
                                     '-s', '1440x1080', 
                                     '-pattern_type', 'glob',
                                     '-i', os.path.join(cam_img_path, '%s_%s_%s_session_%d_cam%d_event_%s_frame_*.jpg' % (marms, date, exp, sNum, cam, event)), 
                                     '-crf', '15',
                                     '-vcodec', 'libx265', 
                                     os.path.join(video_path, '%s_%s_%s_%s_s%d_e%s_cam%d.%s' % (marms, date, start_time, exp, sNum, event, cam, video_ext))])

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--marms", required=True,
     	help="marmoset 4-digit code, e.g. JLTY")
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
    args = vars(ap.parse_args())
    
    session_nums = [int(num) for num in args['session_nums']]
    
    convert_jpg_to_video(args['marms'], 
                         args['dates'], 
                         args['exp_name'], 
                         session_nums, 
                         args['fps'], 
                         args['ncams'], 
                         video_ext = 'avi')