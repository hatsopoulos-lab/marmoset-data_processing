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
    if extraction_dict['dlc_iter'] is not None:
        dlc_cfg['iteration'] = extraction_dict['dlc_iter']
    
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
        
        if extraction_dict['mode'] == 'outliers':
            for cam in cameras:
                single_cam_videos   = [vid for vid in cam_videos if int(re.findall(cam_pattern, os.path.basename(vid))[0].split('_cam')[-1]) == cam]                       
                if isinstance(extraction_dict['bodyparts'], dict):
                    bodyparts = extraction_dict['bodyparts'][f'cam{cam}']
                else:
                    bodyparts = extraction_dict['bodyparts']
                deeplabcut.extract_outlier_frames(dlc_config_path, 
                                                  single_cam_videos, 
                                                  outlieralgorithm='jump', 
                                                  comparisonbodyparts=bodyparts,
                                                  extractionalgorithm=extraction_dict['extract_algo'],
                                                  automatic=True,)
                
                
        elif extraction_dict['mode'] == 'original':
            videos_to_add = [v for v in cam_videos if v not in dlc_cfg["video_sets"]]
            print(videos_to_add)
            deeplabcut.add_new_videos(dlc_config_path,
                                      videos_to_add,
                                      copy_videos=False,
                                      coords=None,
                                      extract_frames=False,)

            deeplabcut.extract_frames(dlc_config_path,
                                      videos_list=cam_videos,
                                      algo=extraction_dict['extract_algo'],
                                      mode='automatic',
                                      userfeedback=False,)
            
    


            
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-d", "--date_dir", required=True, type=str,
    #     help="path to anipose directory. E.g. '/project/nicho/data/marmosets/kinematics_videos/moths/HMMG/2023_04_16/'")
    # ap.add_argument("-p", "--pose_dir", required=True, type=str,
    #     help="name of directory holding .h5 and .pickle files. E.g. 'pose-2d-viterbi_and_autoencoder'")
    # ap.add_argument("-v", "--vid_dir", required=True, type=str,
    #     help="name to directory holding original .avi files. E.g. 'avi_videos'")
    # ap.add_argument("-", "--overwrite", required=True, type=str,
    #     help="True/False whether overwriting training dataset is allowed")
    # ap.add_argument("-m", "--maxiters", required=True, type=int,
    #  	help="number of iterations to stop training")
    # # args = vars(ap.parse_args())
    
    ### MG
    
    # extraction_dict = {'events'         : [    1,    2,    2,    2,    2,    2,    2,    2,    3,    3,    3,    4,    5,    5,    5,    6,    6,    6,    6,    7,    8,    8,   9,    10,   10,   11,   12,   13,   14,   15,   15,   16,   17,   18,   18,   19,   20,   20,   20,   21,   21,   21,   22,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,   32,   33,   34,   37,   38,   39,   39,   39,   40],
    #                   'start_fractions': [ 0.15, 0.03, 0.16, 0.25, 0.45, 0.74, 0.82, 0.89, 0.23, 0.36, 0.71, 0.14,    0, 0.57, 0.80, 0.03, 0.28, 0.47, 0.72, 0.10, 0.01, 0.68, 0.5,  0.04, 0.71, 0.08, 0.29, 0.08, 0.11, 0.02, 0.70, 0.18, 0.08, 0.04, 0.48, 0.05, 0.01, 0.19, 0.41, 0.02, 0.45, 0.68, 0.14, 0.71, 0.23, 0.03, 0.33, 0.06, 0.11, 0.10, 0.17, 0.07, 0.57, 0.18, 0.10, 0.57, 0.04, 0.04, 0.35, 0.56, 0.83, 0.58],
    #                   'stop_fractions' : [ 0.38, 0.09, 0.18, 0.27, 0.46, 0.76, 0.86, 0.96, 0.29, 0.47, 0.84, 0.33, 0.06, 0.67, 0.84, 0.11, 0.30, 0.50, 0.89, 0.30, 0.06, 0.71,   1,  0.16, 0.78, 0.44, 0.46, 0.19, 0.19, 0.07, 0.79, 0.32, 0.25, 0.13, 0.56, 0.33, 0.09, 0.24, 0.44, 0.05, 0.52, 0.81, 0.17, 0.76, 0.33, 0.13,    1, 0.25, 0.26, 0.43, 0.33, 0.28, 0.64, 0.47, 0.30, 0.62, 0.21, 0.15, 0.41, 0.58, 0.85, 0.66],
    #                   'numframes'      : [    0,    4,    4,    4,    4,    4,    4,    4,    4,    4,    4,    0,    0,    0,    0,    4,    4,    4,    4,    0,    0,    0,   4,     4,    4,    6,    2,    2,    4,    6,    6,    0,   10,    2,    2,    0,    2,    2,    2,    2,    2,    2,    0,    0,    0,    4,    0,    2,    2,    2,    2,    6,    0,    2,    0,    0,    0,    2,    4,    4,    4,    6],
    #                   'cameras'        : [[2,4],[1,3],[2,4],[2,4],[1,3],[2,4],[1,3],[2,4],[2,4],[2,4],[2,4],[2,4],[2,4],[1,3],[1,3],[2,4],[2,4],[2,4],[1,3],[2,4],[2,4],[2,4],[2,4],[1,3],[2,4],[1,3],[2,4],[2,4],[2,4],[2,4],[2,4],[2,4],[1,3],[2,4],[2,4],[1,3],[2,4],[1,3],[2,4],[2,4],[2,4],[1,3],[2,4],[2,4],[2,4],[2,4],[2,4],[2,4],[2,4],[2,4],[2,4],[1,3],[2,4],[2,4],[2,4],[2,4],[2,4],[2,4],[1,3],[2,4],[2,4],[1,3]],
    #                   'bodyparts': ['l-wrist'],
    #                   'dlc_iter': 9}
    
    # extraction_dict = {'events'        : [    2,    2,    2,    2,    2,    2,    2,    3,    3,    3,    6,    6,    6,    6,   9,    10,   10,   11,   12,   13,   14,   15,   15,   17,   18,   18,   20,   20,   20,   21,   21,   21,   24,   26,   27,   28,   29,   30,   32,   38,   39,   39,   39,   40],
    #                   'start_fractions': [ 0.03, 0.16, 0.25, 0.45, 0.74, 0.82, 0.89, 0.23, 0.36, 0.71, 0.03, 0.28, 0.47, 0.72, 0.5,  0.04, 0.71, 0.08, 0.29, 0.08, 0.11, 0.02, 0.70, 0.08, 0.04, 0.48, 0.01, 0.19, 0.41, 0.02, 0.45, 0.68, 0.03, 0.06, 0.11, 0.10, 0.17, 0.07, 0.18, 0.04, 0.35, 0.56, 0.83, 0.58],
    #                   'stop_fractions' : [ 0.09, 0.18, 0.27, 0.46, 0.76, 0.86, 0.96, 0.29, 0.47, 0.84, 0.11, 0.30, 0.50, 0.89,   1,  0.16, 0.78, 0.44, 0.46, 0.19, 0.19, 0.07, 0.79, 0.25, 0.13, 0.56, 0.09, 0.24, 0.44, 0.05, 0.52, 0.81, 0.13, 0.25, 0.26, 0.43, 0.33, 0.28, 0.47, 0.15, 0.41, 0.58, 0.85, 0.66],
    #                   'numframes'      : [    4,    4,    4,    4,    4,    4,    4,    4,    4,    4,    4,    4,    4,    4,   4,     4,    4,    6,    2,    2,    4,    6,    6,   10,    2,    2,    2,    2,    2,    2,    2,    2,    4,    2,    2,    2,    2,    6,    2,    2,    4,    4,    4,    6],
    #                   'cameras'        : [[1,3],[2,4],[2,4],[1,3],[2,4],[1,3],[2,4],[2,4],[2,4],[2,4],[2,4],[2,4],[2,4],[1,3],[2,4],[1,3],[2,4],[1,3],[2,4],[2,4],[2,4],[2,4],[2,4],[1,3],[2,4],[2,4],[2,4],[1,3],[2,4],[2,4],[2,4],[1,3],[2,4],[2,4],[2,4],[2,4],[2,4],[1,3],[2,4],[2,4],[1,3],[2,4],[2,4],[1,3]],
    #                   'bodyparts': ['l-wrist'],
    #                   'dlc_iter': 9}

    ### JL_2023_08_03

    # args = {'pose_dir': '/project/nicho/data/marmosets/kinematics_videos/moth/JLTY/2023_08_03/pose-2d-viterbi_and_autoencoder',
    #         'vid_dir' : '/project/nicho/data/marmosets/kinematics_videos/moth/JLTY/2023_08_03/avi_videos',
    #         'dlc_config_path': '/project/nicho/projects/marmosets/dlc_project_files/simple_marmoset_model-Dalton-2023-04-28/config.yaml'}
    
    # extraction_dict = {'events'         :[   17,  30,    39,   40],
    #                    'start_fractions':[ 0.08, 0.07, 0.35, 0.58],
    #                    'stop_fractions' :[ 0.25, 0.28, 0.41, 0.66],
    #                    'numframes'      :[   10,    6,    4,    6],
    #                    'cameras'        :[[2,4],[2,4],[2,4],[2,4]],
    #                    'bodyparts'      :['l-wrist'],
    #                    'dlc_iter'       :9}

    ### JL_2023_08_04

    # args = {'pose_dir': '/project/nicho/data/marmosets/kinematics_videos/moth/JLTY/2023_08_04/pose-2d-viterbi_and_autoencoder',
    #         'vid_dir' : '/project/nicho/data/marmosets/kinematics_videos/moth/JLTY/2023_08_04/avi_videos',
    #         'dlc_config_path': '/project/nicho/projects/marmosets/dlc_project_files/simple_5cams_marmoset_model-Dalton-2024-06-27/config.yaml'}
    
    # # extraction_dict = {'events'          :[     5,    12,    16,    19,    29,    34,    36,    48,],
    # #                     'start_fractions':[  0.00,  0.53,  0.10,  0.40,  0.23,  0.23,  0.30,  0.06,],
    # #                     'stop_fractions' :[  0.45,  0.63,  0.43,  0.61,  0.28,  0.33,  0.45,  0.39,],
    # #                     'numframes'      :[    10,    10,    10,    10,    10,    10,    10,    10,],
    # #                     'cameras'        :[[1,2,],[3,4,],[1,4,],[2,3,],[1,2,],[3,4,],[1,4,],[2,3,],],
    # #                     'bodyparts'      :['l-wrist'],
    # #                     'dlc_iter'       :1,
    # #                     'extract_algo'   :['kmeans', 'uniform'][1],
    # #                     'mode'           :['original', 'outliers'][0],}
    # extraction_dict = {'events'          :[    7,     9,    12,    13,    18,    29,    33,    35,    36,    46,    46,    48,    48,],
    #                     'start_fractions':[0.324, 0.170, 0.603, 0.426, 0.133, 0.220, 0.075, 0.318, 0.347, 0.103, 0.241, 0.102, 0.266,],
    #                     'stop_fractions' :[0.342, 0.286, 0.643, 0.450, 0.229, 0.379, 0.087, 0.360, 0.422, 0.117, 0.262, 0.115, 0.334,],
    #                     'numframes'      :[    3,     3,     3,     3,     5,     3,     5,     3,     5,     5,    7,     10,     5,],
    #                     'cameras'        :[  [2],   [4],   [2],   [4], [2,4],   [2], [2,4],   [4], [2,4], [2,4], [2,4], [2,4], [2,4],],
    #                     'bodyparts'      : dict(cam1=['l-wrist', 'l-shoulder'], 
    #                                             cam2=['l-wrist', 'l-shoulder'], 
    #                                             cam3=['l-wrist', 'l-shoulder'],
    #                                             cam4=['l-wrist', 'l-shoulder'],
    #                                             cam5=['r-wrist', 'l-wrist'],),
    #                     'dlc_iter'       :5,
    #                     'extract_algo'   :['kmeans', 'uniform'][1],
    #                     'mode'           :['original', 'outliers'][0],}  

    ### JL_2023_08_05
    
    args = {'pose_dir': '/project/nicho/data/marmosets/kinematics_videos/moth/JLTY/2023_08_05/pose-2d-viterbi_and_autoencoder',
            'vid_dir' : '/project/nicho/data/marmosets/kinematics_videos/moth/JLTY/2023_08_05/avi_videos',
            'dlc_config_path': '/project/nicho/projects/marmosets/dlc_project_files/simple_5cams_marmoset_model-Dalton-2024-06-27/config.yaml'}
    
    extraction_dict = {'events'          :[   10,    13,    13,    18,    18,   18,   23,   23,    25,    25,    25,    33,    33,],
                        'start_fractions':[ 0.65, 0.023, 0.321, 0.269, 0.277, 0.75, 0.20, 0.31, 0.564, 0.576, 0.596, 0.588, 0.610,],
                        'stop_fractions' :[ 0.68, 0.025, 0.325, 0.272, 0.292, 0.81, 0.23, 0.33, 0.570, 0.578, 0.602, 0.602, 0.615,],
                        'numframes'      :[    5,     3,     3,     3,     3,    5,    5,    5,     2,     3,     3,     3,     3,],
                        'cameras'        :[[2,4], [2,4], [2,4], [2,4], [2,4],[2,4],[2,4],[2,4], [2,4], [2,4], [2,4], [2,4], [2,4],],
                        'bodyparts'      :['l-wrist'],
                        'dlc_iter'       :5,
                        'extract_algo'   :['kmeans', 'uniform'][0],
                        'mode'           :['original', 'outliers'][0],}

    pose_dir = args['pose_dir']
    if pose_dir[-1] == '/':
        pose_dir = pose_dir[:-1]
    
    if extraction_dict['mode'] == 'outliers':
        with open(os.path.join(os.path.split(pose_dir)[0], 'scorer_info.txt')) as f:
            scorer = f.readlines()[0]
        scorer = scorer.split('filtered')[-1]
        scorer = scorer.split('_meta')[0]
    
        revert_and_copy_pose_filenames(pose_dir,
                                       args['vid_dir'],
                                       scorer)
    
    extract_outlier_frames(args['vid_dir'], args['dlc_config_path'], extraction_dict)
