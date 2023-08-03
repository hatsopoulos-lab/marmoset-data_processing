# -*- coding: utf-8 -*-
"""
Created on June 07 2022

@author: Dalton
"""

import glob
import os
import re
import argparse
import shutil  
import pandas as pd
import deeplabcut

def revert_and_copy_pose_filenames(pose_dir, vid_dir, scorer):
    
    pose_files = sorted(glob.glob(os.path.join(pose_dir, '*')))
    for f in pose_files:
        base = os.path.basename(f)
        base, ext = os.path.splitext(base)
        
        new_file = base + scorer + ext
        new_file = os.path.join(vid_dir, new_file)
        # shutil.copy(f, new_file)

        print('\n old file: %s \n new file: %s \n' % (f, new_file))
        
        df = pd.read_hdf(f)
        df.columns = df.columns.set_levels([scorer], level=0)
        df.to_hdf(new_file, key="df_with_missing", mode="w")
        
def extract_outlier_frames(video_path, dlc_config_path, extraction_dict):
    cam_pattern = re.compile('_cam[0-9]{1}')
    
    dlc_cfg=deeplabcut.auxiliaryfunctions.read_config(dlc_config_path)
    
    for event, start, stop, nframes, cameras in zip(extraction_dict['events'],
                                                    extraction_dict['start_fractions'],
                                                    extraction_dict['stop_fractions'],
                                                    extraction_dict['numframes'],
                                                    extraction_dict['cameras']):
        
        event_videos = sorted(glob.glob(os.path.join(video_path, '*_e%s_*.avi' % str(event).zfill(3))))
        cam_videos   = [vid for vid in event_videos if int(re.findall(cam_pattern, os.path.basename(vid))[0].split('_cam')[-1]) in cameras]                       
         
        print((event, cam_videos))                     
        dlc_cfg['start'] = start
        dlc_cfg['stop']  = stop
        dlc_cfg['numframes2pick'] = nframes
        deeplabcut.auxiliaryfunctions.write_config(dlc_config_path, dlc_cfg)
        
        deeplabcut.extract_outlier_frames(dlc_config_path, 
                                          cam_videos, 
                                          outlieralgorithm='jump', 
                                          comparisonbodyparts=extraction_dict['bodyparts'], 
                                          automatic=True)
    
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--pose_dir", required=True, type=str,
    #     help="path to directory holding .h5 and .pickle files. E.g. '/project/nicho/data/marmosets/kinematics_videos/test/TYJL/2022_06_17/pose-2d'")
    # ap.add_argument("-v", "--vid_dir", required=True, type=str,
    #     help="path to directory holding original .avi files. E.g. '/project/nicho/data/marmosets/kinematics_videos/test/TYJL/2022_06_17/avi_videos'")

    # args = vars(ap.parse_args())
    
    args = {'pose_dir': '/project/nicho/data/marmosets/kinematics_videos/moths/HMMG/2023_04_16/pose-2d-viterbi_and_autoencoder',
            'vid_dir' : '/project/nicho/data/marmosets/kinematics_videos/moths/HMMG/2023_04_16/avi_videos',
            'dlc_config_path': '/project/nicho/projects/marmosets/dlc_project_files/simple_marmoset_model-Dalton-2023-04-28/config.yaml'}
    
    extraction_dict = {'events'         : [    1,     2,    25,    33,    33,    33,    39,    40,    43,    44,    44,    45,    52],
                       'start_fractions': [ 0.51,  0.59,  0.72,  0.12,  0.71,  0.92,  0.38,  0.09,  0.41,  0.11,   0.3,  0.32,  0.21],
                       'stop_fractions' : [ 0.56,  0.71,  0.81,  0.17,  0.88,  0.96,     1,   0.3,   0.7,  0.22,  0.66,  0.64,  0.26],
                       'numframes'      : [    5,     5,     5,    15,    15,    15,     5,     5,     5,    10,    25,    10,    10],
                       'cameras'        : [[1,3], [1,3], [1,3], [1,3], [1,3], [1,3], [1,3], [1,3], [1,3], [1,3], [1,3], [1,3], [1,3]],
                       'bodyparts': ['r-wrist']}
    
    pose_dir = args['pose_dir']
    if pose_dir[-1] == '/':
        pose_dir = pose_dir[:-1]
    with open(os.path.join(os.path.split(pose_dir)[0], 'scorer_info.txt')) as f:
        scorer = f.readlines()[0]
    
    scorer = scorer.split('filtered')[-1]
    scorer = scorer.split('_meta')[0]
    
    revert_and_copy_pose_filenames(pose_dir,
                                   args['vid_dir'],
                                   scorer)
    
    
    extract_outlier_frames(args['vid_dir'], args['dlc_config_path'], extraction_dict)
