# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:21:33 2021

@author: Dalton
"""

import argparse
import numpy as np
import shutil
import re
import pandas as pd
import os
from pathlib import Path

def insert_labels(anipose_args):  

    labeled_data_path = Path(anipose_args['projectpath']) / 'labeled-data'     

    date_dir = Path(anipose_args['date_dir'])
    task_path = date_dir / 'temp_anipose_processing' / str(anipose_args['task_id'])
    os.makedirs(task_path / 'pose-2d-labels-inserted', exist_ok=True)     
    os.chdir(task_path)

    h5_files = list((task_path / 'pose-2d-unfiltered').glob('*.h5'))
    for f in h5_files:        
        pose   = pd.read_hdf(f)
        labels_files = list((labeled_data_path / f.stem).glob('CollectedData*.h5'))
        if len(labels_files) > 0:
            labels = pd.read_hdf(labels_files[0])

            dlc_scorer = pose.columns[0][0]
            for col in labels.columns:
                pose_col = (dlc_scorer, col[1], col[2])
                for row_index, label_pos in labels[col].items():
                    if ~np.isnan(label_pos):
                        frame = int(row_index[-1].split('img')[-1].split('.png')[0])
                        pose_dtype = pose.loc[frame, pose_col].dtype
                        pose.loc[frame, pose_col] = np.array(label_pos).astype(pose_dtype)
                        if pose_col[-1] == 'y':
                            like_col = (pose_col[0], pose_col[1], 'likelihood')
                            # like_dtype = pose.loc[frame, like_col].dtype
                            pose.loc[frame, like_col] = 1.0
                    #     print(col, row_index[-1], frame, label_pos)
                    # else:
                    #     print('isnan')

        pose.to_hdf(task_path / 'pose-2d-labels-inserted' / f.name, 
                    "df_with_missing", format="table", mode="w")


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
    debugging = True
    
    if not debugging:
    
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dlc_path", required=True, type=str,
            help="path to dlc project. E.g. '/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26'")
        ap.add_argument("-a", "--anipose_path", required=True, type=str,
            help="path to anipose project. E.g. '/project/nicho/data/marmosets/test'")
        ap.add_argument("-i", "--dlc_iter", required=True, type=str,
            help="dlc project iteration to use. If 'None', use the iteration already in config.yaml")
        ap.add_argument("-f", "--train_frac", required=True, type=str,
            help="dlc training fraction to use. If 'None', use the training fraction already in config.yaml")
        ap.add_argument("-s", "--snap_idx", required=True, type=int,
            help="dlc project snapshot index to use. '-1' will use the last saved snapshot")
        ap.add_argument("-p", "--parameter_set", required=True, type=str,
            help="parameter set to use. E.g. marms or bci")
        ap.add_argument("-v", "--extra_vars", required=True, nargs='+', type=str,
            help="Additional variables to modify functionality: [only3D, label_videos]. Input is a list (array in bash) of True/False, e.g. (False True)")
        ap.add_argument("-dt", "--dates", nargs='+', required=True, type=str,
         	help="date(s) of videos to run thru anipose (can have multiple entries separated by spaces)")
        ap.add_argument("-n", "--ncams", required=True, type=int,
         	help="number of cameras")
        args = vars(ap.parse_args())

    else:
        args = {'dlc_path'     : '/project/nicho/projects/marmosets/dlc_project_files/simple_5cams_marmoset_model-Dalton-2024-06-27',
                'anipose_path' : '/project/nicho/data/marmosets/kinematics_videos/moth/JLTY',
                'dlc_iter'     : 'None',
                'train_frac'   : 'None',
                'snap_idx'     : -1,
                'parameter_set': 'marmoset',
                'extra_vars'   : ['False', 'True'],
                'dates'        : ['2023_08_04'],
                'ncams'        : 5}

    if not debugging:
        task_id   = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        n_tasks   = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
        last_task = int(os.getenv('SLURM_ARRAY_TASK_MAX'))
    else:
        task_id = 0
        n_tasks = 1
        last_task = task_id
        
    print(type(args['dlc_iter']), args['dlc_iter'], args['dlc_iter'] is None)
    print(type(args['train_frac']), args['train_frac'], args['train_frac'] is None)
    print(type(args['extra_vars']), args['extra_vars'])

    if 'pytorch' not in args.keys():
        pytorch = False
    else:
        pytorch    = convert_string_inputs_to_int_float_or_bool(args['pytorch'])
        
    iteration  = convert_string_inputs_to_int_float_or_bool(args['dlc_iter'])
    train_frac = convert_string_inputs_to_int_float_or_bool(args['train_frac'])
    extra_vars = convert_string_inputs_to_int_float_or_bool(args['extra_vars'])
    
    if 'copy_up_to' in args.keys():
        copy_up_to = convert_string_inputs_to_int_float_or_bool(args['copy_up_to'])
    else:
        copy_up_to = None

    epPattern = re.compile('_e[0-9]{3}') 
    epPattern_backup = re.compile('_event[0-9]{3}')    
    for date in args['dates']:
        ddir = Path(args['anipose_path']) / date
        videos = sorted(list((ddir / 'avi_videos').glob('*.avi')))
        
        try:
            events = np.unique([int(re.findall(epPattern,        vid.name)[0].split('_e')[-1]) for vid in videos])
        except:
            events = np.unique([int(re.findall(epPattern_backup, vid.name)[0].split('_event')[-1]) for vid in videos])
            
        cam1_videos = [vidpath for vidpath in videos if 'cam1' in vidpath.name]
        event_video_sizes = [round(os.stat(vidpath).st_size/(1024**2)) for event, vidpath in zip(events, cam1_videos)]
        
        event_video_size_df = pd.DataFrame(data=zip(events, event_video_sizes),
                                           columns=['event', 'video_size'])
        event_video_size_df.sort_values(by='video_size', ascending=False, inplace=True, ignore_index=True)
        
        all_task_events_lists = [[] for tmp in range(n_tasks)]
        all_task_total_sizes  = [0 for tmp in range(n_tasks)]
        assigned_events = []
        # assign first set_of_events
        for idx, tmp_info in event_video_size_df.iterrows():
            if idx < n_tasks:
                assign_idx = idx
            else:
                assign_idx = np.argmin(all_task_total_sizes)
            all_task_events_lists[assign_idx].append(tmp_info['event'])
            all_task_total_sizes [assign_idx] += (tmp_info['video_size'] * n_tasks)
            assigned_events.append(tmp_info['event'])
        
        task_events = all_task_events_lists[task_id]
        try:
            task_videos = [vid for vid in videos if int(re.findall(epPattern,        vid.name)[0].split('_e')[-1]) in task_events]   
        except:
            task_videos = [vid for vid in videos if int(re.findall(epPattern_backup, vid.name)[0].split('_event')[-1]) in task_events]   

        anipose_args = {'projectpath'     : args['dlc_path'],
                        'aniposepath'     : args['anipose_path'],
                        'iteration'       : iteration,
                        'train_frac'      : train_frac,
                        'snap_idx'        : args['snap_idx'],
                        'parameter_set'   : args['parameter_set'],
                        'only_3D'         : extra_vars[0],
                        'label_videos'    : extra_vars[1],
                        'task_id'         : task_id,
                        'n_task'          : n_tasks,
                        'task_videos'     : task_videos,
                        'date_dir'        : ddir,
                        'ncams'           : args['ncams'],
                        'copy_up_to'      : copy_up_to,
                        'pytorch'         : pytorch}
    
        for key, val in anipose_args.items():
            print(key, ' : ', val, flush=True)
    
        insert_labels(anipose_args)
                                
