# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:21:33 2021

@author: Dalton
"""

import subprocess
import os
import deeplabcut
import toml
import argparse
import numpy as np
import shutil
import time
import re
import pandas as pd
from pathlib import Path
from itertools import product

def edit_anipose_params(config_data, config_file, categories, keys, values):
    print(type(categories), categories)
    print(type(keys), keys)
    print(type(values), values)
    print('', flush=True)
    for cat, key, val in zip(categories, keys, values):
        print(cat, key, val)
        if key is not None:
            config_data[cat][key] = val
        else:
            config_data[cat]      = val
    
    with open(config_file, 'w') as f:
        toml.dump(config_data, f)
        
    return

def count_files_in_folders(task_path, folders):
    
    task_paths = task_path.parent.glob('*') 
    
    filecount = 0
    for task_dir, fold in product(task_paths, folders):
        filecount += len(list((task_dir / fold).glob('*')))

    return filecount           

def compute_pose_with_anipose(anipose_args):  
    
    param_category = ['filter', 'filter', 'triangulation', 'triangulation', 'triangulation', 'triangulation', 'triangulation']

    if 'marm' in anipose_args['parameter_set'].lower():
        test_params = {'offset_threshold'       : 20, 
                       'n_back'                 : 5, 
                       'scale_smooth'           : 4, 
                       'scale_length'           : 2,
                       'scale_length_weak'      : 0,
                       'reproj_error_threshold' : 8,
                       'score_threshold'        : 0.3} 
    elif 'bci' in anipose_args['parameter_set'].lower():
        test_params = {'offset_threshold'       : 20, 
                       'n_back'                 : 5, 
                       'scale_smooth'           : 4, 
                       'scale_length'           : 6,
                       'scale_length_weak'      : 0,
                       'reproj_error_threshold' : 8,
                       'score_threshold'        : 0.3}
 
    param_names  = list(test_params)
    param_values = list(test_params.values())

    param_names.extend([None, None])
    param_values.extend([anipose_args['projectpath'], 0])
    param_category.extend(['model_folder', 'nesting'])
    
    date_dir = Path(anipose_args['date_dir'])
    
    if anipose_args['task_id'] == 0:
        dlc_config=Path(anipose_args['projectpath']) / 'config.yaml'
        print(dlc_config, flush=True)
        dlc_cfg=deeplabcut.auxiliaryfunctions.read_config(dlc_config)
        
        original_iteration=dlc_cfg['iteration']
        original_snapshotindex = dlc_cfg['snapshotindex']
        original_train_fraction = dlc_cfg['TrainingFraction']
        
        if anipose_args['iteration'] is not None:
            dlc_cfg['iteration'] = anipose_args['iteration']
        if anipose_args['snap_idx'] is not None:
            dlc_cfg['snapshotindex'] = anipose_args['snap_idx']
        if anipose_args['train_frac'] is not None:
            dlc_cfg['TrainingFraction'] = [anipose_args['train_frac']]
        
        deeplabcut.auxiliaryfunctions.write_config(dlc_config, dlc_cfg)
    
    # copy config.toml file into date directory and edit parameters, then calibrate cameras
    ani_config   = date_dir                          / 'config.toml'
    ani_template = Path(anipose_args['aniposepath']) / 'config.toml'
    shutil.copy(ani_template, ani_config)    
    
    os.chdir(date_dir)
    
    ani_cfg_data = toml.load(ani_config) 
    edit_anipose_params(ani_cfg_data, ani_config, 
                        param_category, 
                        param_names, 
                        param_values)

    if anipose_args['task_id'] == 0:
        print('\n\nMade it to the calibrate line!!!\n\n', flush=True)
        subprocess.call(['anipose', 'calibrate'])
    else:
        task_0_calib_path = date_dir / 'temp_anipose_processing' / '0' / 'calibration' / 'calibration.toml'
        ct = 0
        print((task_0_calib_path, task_0_calib_path.is_file(), ct), flush=True) 
        while not task_0_calib_path.is_file():
            time.sleep(5)
            ct+=1
            print((task_0_calib_path, task_0_calib_path.is_file(), ct), flush=True) 


        print('passed this point')
    
    # copy config file and calibration results temp_anipose_processing/TASKID folders
    task_path = date_dir / 'temp_anipose_processing' / str(anipose_args['task_id'])
    os.makedirs(task_path / 'calibration', exist_ok=True)
    os.makedirs(task_path / 'avi_videos' , exist_ok=True)
    shutil.copy(ani_config, task_path)
    shutil.copy(date_dir / 'calibration' / 'calibration.toml', task_path / 'calibration')
    
    # move into directory for task processing
    os.chdir(task_path)
    
    task_ani_config = task_path / 'config.toml'
    task_ani_cfg_data = toml.load(task_ani_config) 
    
    # copy the videos and for this task into the TASK_ID folder
    for vidpath in anipose_args['task_videos']:
        shutil.copy(vidpath, task_path / 'avi_videos')
    
    # Copy additional files into the TASK_ID folder, up to the "copy_up_to" input argument.
    # This allows for skippng steps that have already been completed if a job is canceled for some undesired reason and 
    # you want to restart without re-doing everything (particularly the initial inference step by DLC)
    if copy_up_to is not None:
        folders_in_order = ['pose-2d-unfiltered', 'pose-2d-viterbi', 'pose-2d-viterbi_and_autoencoder',
                            'pose-3d', 'pose-2d-proj', 'videos-2d-proj', 'videos-labeled-filtered']
        for fold in folders_in_order:
            os.makedirs(task_path / fold, exist_ok=True)
            for vid in anipose_args['task_videos']:
                vidpath = Path(vid)
                for src_file in (vidpath.parent.parent / fold).glob(f'{vidpath.stem}*'):
                    shutil.copy(src_file, task_path / fold)
            if fold == anipose_args['copy_up_to']:
                break

    print('\n' + os.getcwd(), flush=True)
    if not anipose_args['only_3D']:        

        print(task_ani_cfg_data, flush=True)
        print(task_ani_config, flush=True)
        edit_anipose_params(task_ani_cfg_data, task_ani_config, 
                            ['filter' , 'pipeline'       , 'pipeline'          ], 
                            ['type'   , 'pose_2d_filter' , 'pose_2d'           ], 
                            ['viterbi', 'pose-2d-viterbi', 'pose-2d-unfiltered'])

        subprocess.call(['anipose', 'analyze'])
        subprocess.call(['anipose', 'filter'])     
    
        edit_anipose_params(task_ani_cfg_data, task_ani_config, 
                            ['filter'     , 'pipeline'                       , 'pipeline'       ], 
                            ['type'       , 'pose_2d_filter'                 , 'pose_2d'        ], 
                            ['autoencoder', 'pose-2d-viterbi_and_autoencoder', 'pose-2d-viterbi'])
        
        edit_anipose_params(ani_cfg_data, ani_config, 
                            ['filter'     , 'pipeline'                       , 'pipeline'       ], 
                            ['type'       , 'pose_2d_filter'                 , 'pose_2d'        ], 
                            ['autoencoder', 'pose-2d-viterbi_and_autoencoder', 'pose-2d-viterbi'])
        
        # copy files to date folder, train autoencoder and apply it in the primary folder. Then triangulate in primary folder 
        for src_file in sorted(list((task_path / 'pose-2d-viterbi').glob('*'))):
            dst_file = date_dir / 'pose-2d-viterbi' / src_file.name
            shutil.copy(src_file, dst_file)
        
        if anipose_args['task_id'] == 0:
            os.chdir(date_dir)
            while len(list((date_dir / 'pose-2d-viterbi').glob('*'))) < len(list((date_dir / 'avi_videos').glob('*.avi'))):
                time.sleep(10)
            subprocess.call(['anipose', 'train-autoencoder'])  
            shutil.copy(date_dir / 'autoencoder.pickle', task_path)
        else:
            task_0_path = date_dir / 'temp_anipose_processing' / '0'
            while not (task_0_path / 'autoencoder.pickle').exists():
                time.sleep(5)
            shutil.copy(date_dir / 'autoencoder.pickle', task_path)
            
        os.chdir(task_path)    
        
        subprocess.call(['anipose', 'filter'])
        subprocess.call(['anipose', 'triangulate']) 
        subprocess.call(['anipose', 'project-2d']) 
 
        if anipose_args['label_videos']:
            subprocess.call(['anipose', 'label-2d-proj']) 
            subprocess.call(['anipose', 'label-2d-filter']) 
            subprocess.call(['anipose', 'label-3d'])
        
        folders_with_new_info = ['pose-2d-unfiltered',
                                 'pose-2d-viterbi_and_autoencoder',
                                 'pose-2d-proj',
                                 'pose-3d',
                                 'videos-labeled-filtered',
                                 'videos-2d-proj',
                                 'videos-3d']
                
    else:
        
        edit_anipose_params(task_ani_cfg_data, task_ani_config, 
                            ['pipeline'                       , 'pipeline'       ], 
                            ['pose_2d_filter'                 , 'pose_2d'        ], 
                            ['pose-2d-viterbi_and_autoencoder', 'pose-2d-viterbi'])
            
        # copy filtered pose files to temp_anipose folders
        os.makedirs(task_path / 'pose-2d-viterbi_and_autoencoder', exist_ok = True)
        for vidpath in anipose_args['task_videos']:
            vidpath = Path(vidpath)
            pose_path = vidpath.parent.parent / 'pose-2d-viterbi_and_autoencoder' / vidpath.with_suffix('.h5').name
            shutil.copy(pose_path, task_path / 'pose-2d-viterbi_and_autoencoder') 
            
        subprocess.call(['anipose', 'triangulate']) 
        subprocess.call(['anipose', 'project-2d'])           
            
        if anipose_args['label_videos']:
            subprocess.call(['anipose', 'label-2d-proj']) 
            subprocess.call(['anipose', 'label-2d-filter']) 
            
        folders_with_new_info = ['pose-2d-proj',
                                 'pose-3d',
                                 'videos-labeled-filtered',
                                 'videos-2d-proj',]    
                
    # move all new files back to primary anipose folder
    for folder_name in folders_with_new_info:
        src_dir = task_path / folder_name
        if src_dir.is_dir():
            dst_dir = date_dir / folder_name
            os.makedirs(dst_dir, exist_ok=True)
            print('moving files from \n%s to \n%s' % (src_dir, dst_dir))
            for f in src_dir.glob('*'):
                shutil.move(f, dst_dir / f.name)
    
    if anipose_args['task_id'] == last_task:
        shutil.move(task_path / 'scorer_info.txt', date_dir / 'scorer_info.txt')
            
    if anipose_args['task_id'] == last_task:
        filecount = count_files_in_folders(task_path, folders_with_new_info)
        while filecount > 0:
            print(f'File count = {filecount}. Waiting 5 minutes, then checking if all files have been moved from temp_anipose_processing/ to parent')
            time.sleep(60*5)
            filecount = count_files_in_folders(task_path, folders_with_new_info)
        print('\nAbout to remove "temp_anipose_processing"')
        
        # shutil.rmtree(date_dir / 'temp_anipose_processing', ignore_errors=True)
        for task_dir in (date_dir / 'temp_anipose_processing').glob('*'): 
            shutil.rmtree(task_dir / 'avi_videos')
        os.removedirs(date_dir / 'temp_anipose_processing')

        print("resetting snapshotindex and iteration")
        dlc_cfg['iteration'] = original_iteration
        dlc_cfg['snapshotindex'] = original_snapshotindex
        dlc_cfg['TrainingFraction'] = original_train_fraction
        deeplabcut.auxiliaryfunctions.write_config(dlc_config, dlc_cfg)
    
    # print('resetting anipose pipeline variables in %s' % ani_config)
    # ani_cfg_data['pipeline']['pose_2d_filter'] = 'pose-2d-viterbi'
    # ani_cfg_data['pipeline']['pose_2d'] = 'pose-2d-unfiltered'
    # with open(ani_config, 'w') as f:
    #     toml.dump(ani_cfg_data, f)

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
    ap.add_argument("-c", "--copy_up_to", required=False, type=str,
        help="last folder to copy from the base date folder for anipose to 'temp_anipose_processing', \nallowing some steps to be skipped in individual tasks")
    args = vars(ap.parse_args())

    print('\n\n Beginning anipose processing at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)

    print(type(args['dlc_iter']), args['dlc_iter'], args['dlc_iter'] is None)
    print(type(args['train_frac']), args['train_frac'], args['train_frac'] is None)
    print(type(args['extra_vars']), args['extra_vars'])
    
    task_id   = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    n_tasks   = 5 #int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    last_task = 4 #int(os.getenv('SLURM_ARRAY_TASK_MAX'))

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
                        'copy_up_to'      : copy_up_to}
    
        for key, val in anipose_args.items():
            print(key, ' : ', val, flush=True)
        
        if task_id == 0:
            for folder_name in ['pose-2d-unfiltered', 'pose-2d-viterbi', 
                                'pose-2d-viterbi_and_autoencoder', 'pose-3d', 
                                'videos-labeled-filtered', 'pose-2d-proj', 
                                'videos-2d-proj', 'videos-3d']:
                os.makedirs(Path(anipose_args['date_dir']) / folder_name, exist_ok=True)
    
        compute_pose_with_anipose(anipose_args)
                                