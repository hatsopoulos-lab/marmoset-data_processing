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
import os
import subprocess
import argparse
import time

class paths: 
    processing_code   = '/project/nicho/projects/marmosets/code_database/data_processing/kinematics/video_processing'    

def create_calib_videos_if_nonexistent(anipose_path,
                                       transpose,
                                       calib_name = None,
                                       ncams = 2,
                                       video_ext = 'avi'):

    dates = sorted(glob.glob(os.path.join(anipose_path, '20*')))
    
    print(dates)
    
    for date in dates:
        calibration_path = os.path.join(date, 'calibration')
        print('there are %d files in calibration for %s' % (len(glob.glob(os.path.join(calibration_path, '*'))), date) )
        if os.path.exists(calibration_path) and len(glob.glob(os.path.join(calibration_path, '*'))) > 0:
            continue
        
        print('made it here')
        try:
            calib_image_path = glob.glob(os.path.join(date, '*calib*'))
            calib_image_path = [imgpath for imgpath in calib_image_path 
                                if len(glob.glob(os.path.join(imgpath, '*'))) > 0
                                and len(glob.glob(os.path.join(imgpath, '*.%s' % video_ext))) == 0][0]
            print(calib_image_path)
        except:
            print('\nno calib folder for %s \n' % date)
            continue
        
        os.makedirs(calibration_path, exist_ok=True)
                
        for cam, trans in zip(range(1, ncams+1), transpose):
        
            calib_cam_path = os.path.join(calib_image_path, 'cam%d' % cam)
            
            calib_name_pattern = os.path.basename(glob.glob(os.path.join(calib_cam_path, '*'))[0]).split('_frame')[0]
            outvidfile = os.path.join(calibration_path, '%s.%s' % (calib_name_pattern, video_ext))
            if os.path.exists(outvidfile):
                print('%s already exists - skipping' % outvidfile, flush=True)
            else:
                print('Creating %s' % outvidfile, flush=True)
            
            if int(trans) == -1: 
                subprocess.call(['ffmpeg', 
                                 '-r', '20', 
                                 '-f', 'image2', 
                                 '-s', '1440x1080', 
                                 '-pattern_type', 'glob',
                                 '-i', os.path.join(calib_cam_path, '*frame_*.jpg'), 
                                 '-crf', '15',
                                 '-vcodec', 'libx264', 
                                 outvidfile])
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
                                 outvidfile])

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--anipose_path", required=True, type=str,
        help="path to anipose project. E.g. '/project/nicho/data/marmosets/test'")
    ap.add_argument("-t", "--video_transpose", nargs='+', required=True, type=int,
     	help="video transpose codes (see ffmpeg docs for details). Use '-1' for no transpose. E.g. '2 2 1 1' ")
    # ap.add_argument("-c", "--calib_name", required=True, type=str,
    #  	help="name of calibration folder (e.g. 'calib', 'pre_calib', 'None' ")
    ap.add_argument("-n", "--ncams", required=True, type=int,
     	help="number of cameras")
    args = vars(ap.parse_args())
    
    print('\n\n Beginning jpg2avi code at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
    
    # session_nums = [int(num) for num in args['session_nums']]
    # calib_name = args['calib_name']
    # if calib_name == 'None':
    #     calib_name = None
        
    transpose = []
    for trans in args['video_transpose']:
        if trans in [0, 1, 2, 3]:
            transpose.append(trans)
        else:
            transpose.append(-1)

    create_calib_videos_if_nonexistent(args['anipose_path'],
                                       transpose = transpose,
                                       ncams = args['ncams'],
                                       video_ext = 'avi')

