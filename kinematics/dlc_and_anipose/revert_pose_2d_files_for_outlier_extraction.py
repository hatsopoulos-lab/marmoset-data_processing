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
import argparse
import shutil

class paths: 
    processing_code   = '/project/nicho/projects/marmosets/code_database/data_processing/kinematics'    

def revert_and_copy_pose_filenames(pose_dir, vid_dir, scorer):
    
    pose_files = sorted(glob.glob(os.path.join(pose_dir, '*')))
    for f in pose_files:
        base = os.path.basename(f)
        base, ext = os.path.splitext(base)
        
        new_file = base + scorer + ext
        new_file = os.path.join(vid_dir, new_file)
        shutil.copy(f, new_file)
        
    
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pose_dir", required=True,
        help="path to directory holding .h5 and .pickle files. E.g. '/project/nicho/data/marmosets/kinematics_videos/test/TYJL/2022_06_17/pose-2d'")
    ap.add_argument("-v", "--vid_dir", required=True,
        help="path to directory holding origina .avi files. E.g. '/project/nicho/data/marmosets/kinematics_videos/test/TYJL/2022_06_17/avi_videos'")

    args = vars(ap.parse_args())
    
    pose_dir = args['pose_dir']
    if pose_dir[-1] == '/':
        pose_dir = pose_dir[:-1]
    with open(os.path.join(os.path.split(pose_dir)[0], 'scorer_info.txt')) as f:
        scorer = f.readlines()[0]
    
    revert_and_copy_pose_filenames(pose_dir,
                                   args['vid_dir'],
                                   scorer)
