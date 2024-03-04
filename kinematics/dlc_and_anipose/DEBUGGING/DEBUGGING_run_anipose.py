# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:21:33 2021

@author: Dalton
"""

import subprocess
import os
# import deeplabcut
import toml
import argparse
import numpy as np
import glob
import shutil
import time
import re
import pandas as pd
from os.path import join as pjoin

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

def compute_pose_with_anipose(anipose_args):  

    aniposepath = anipose_args['aniposepath']
    
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
    
    if anipose_args['task_id'] == 0:
        dlc_config=pjoin(anipose_args['projectpath'],'config.yaml')
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
    ani_config = pjoin(anipose_args['date_dir'], 'config.toml')
    ani_template = pjoin(aniposepath, 'config.toml')
    shutil.copy(ani_template, ani_config)    
    
    os.chdir(anipose_args['date_dir'])
    
    ani_cfg_data = toml.load(ani_config) 
    edit_anipose_params(ani_cfg_data, ani_config, 
                        param_category, 
                        param_names, 
                        param_values)

    if anipose_args['task_id'] == 0:
        print('\n\nMade it to the calibrate line!!!\n\n', flush=True)
        subprocess.call(['anipose', 'calibrate'])
    else:
        task_0_calib_path = pjoin(anipose_args['date_dir'], 'temp_anipose_processing', '0', 'calibration', 'calibration.toml')
        ct = 0
        print((task_0_calib_path, os.path.isfile(task_0_calib_path), ct), flush=True) 
        while not os.path.isfile(task_0_calib_path):
            time.sleep(5)
            ct+=1
            print((task_0_calib_path, os.path.isfile(task_0_calib_path), ct), flush=True) 


        print('passed this point')
    
    # copy config file and alibration results temp_anipose_processing/TASKID folders
    task_path = pjoin(anipose_args['date_dir'], 'temp_anipose_processing', str(anipose_args['task_id']))
    os.makedirs(pjoin(task_path, 'calibration'), exist_ok=True)
    os.makedirs(pjoin(task_path, 'avi_videos') , exist_ok=True)
    shutil.copy(ani_config, task_path)
    shutil.copy(pjoin(anipose_args['date_dir'], 'calibration', 'calibration.toml'), pjoin(task_path, 'calibration'))
    
    # move into directory for task processing
    os.chdir(task_path)
    
    task_ani_config = pjoin(task_path, 'config.toml')
    task_ani_cfg_data = toml.load(task_ani_config) 
    
    print('\n' + os.getcwd(), flush=True)
    if not anipose_args['only_3D']:        

        # copy the videos for this task into the TASK_ID folder
        for vidpath in anipose_args['task_videos']:
            shutil.copy(vidpath, pjoin(task_path, 'avi_videos'))        

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
        for src_file in sorted(glob.glob(pjoin(task_path, 'pose-2d-viterbi', '*'))):
            dst_file = pjoin(anipose_args['date_dir'], 'pose-2d-viterbi', os.path.basename(src_file))
            shutil.copy(src_file, dst_file)
        
        if anipose_args['task_id'] == 0:
            os.chdir(anipose_args['date_dir'])
            while len(glob.glob(pjoin(anipose_args['date_dir'], 'pose-2d-viterbi', '*'))) < len(glob.glob(pjoin(anipose_args['date_dir'], 'avi_videos', '*.avi'))):
                time.sleep(10)
            subprocess.call(['anipose', 'train-autoencoder'])  
            shutil.copy(pjoin(anipose_args['date_dir'], 'autoencoder.pickle'), task_path)
        else:
            task_0_path = pjoin(anipose_args['date_dir'], 'temp_anipose_processing', '0')
            while not os.path.isfile(pjoin(task_0_path, 'autoencoder.pickle')):
                time.sleep(5)
            shutil.copy(pjoin(anipose_args['date_dir'], 'autoencoder.pickle'), task_path)
                        
        # os.makedirs(pjoin(task_path, 'pose-2d-viterbi_and_autoencoder'), exist_ok = True)
        # os.makedirs(pjoin(task_path, 'pose-2d-proj'), exist_ok = True)
        # for vidpath in anipose_args['task_videos']:
        #     base_path, filename = os.path.split(vidpath)
        #     base_path = os.path.dirname(base_path)
        #     filename = os.path.splitext(filename)[0] + '.h5'
           
        #     pose_path = pjoin(base_path, 'pose-2d-viterbi_and_autoencoder', filename)
        #     shutil.copy(pose_path, pjoin(task_path, 'pose-2d-viterbi_and_autoencoder')) 

        #     pose_path = pjoin(base_path, 'pose-2d-proj', filename)
        #     shutil.copy(pose_path, pjoin(task_path, 'pose-2d-proj')) 
            
        os.chdir(task_path)    
        
        subprocess.call(['anipose', 'filter'])
        subprocess.call(['anipose', 'triangulate']) 
        subprocess.call(['anipose', 'project-2d']) 
 
        if anipose_args['label_videos']:
            subprocess.call(['anipose', 'label-2d-proj']) 
            subprocess.call(['anipose', 'label-2d-filter']) 
        
        folders_with_new_info = ['pose-2d-unfiltered',
                                 'pose-2d-viterbi_and_autoencoder',
                                 'pose-2d-proj',
                                 'pose-3d',
                                 'videos-labeled-filtered',
                                 'videos-2d-proj']
                
    else:
        
        edit_anipose_params(task_ani_cfg_data, task_ani_config, 
                            ['pipeline'                       , 'pipeline'       ], 
                            ['pose_2d_filter'                 , 'pose_2d'        ], 
                            ['pose-2d-viterbi_and_autoencoder', 'pose-2d-viterbi'])
            
        # copy filtered pose files to temp_anipose folders
        os.makedirs(pjoin(task_path, 'pose-2d-viterbi_and_autoencoder'), exist_ok = True)
        for vidpath in anipose_args['task_videos']:
            base_path, filename = os.path.split(vidpath)
            base_path = os.path.dirname(base_path)
            filename = os.path.splitext(filename)[0] + '.h5'
            pose_path = pjoin(base_path, 'pose-2d-viterbi_and_autoencoder', filename)
            shutil.copy(pose_path, pjoin(task_path, 'pose-2d-viterbi_and_autoencoder')) 
            
        subprocess.call(['anipose', 'triangulate']) 
        subprocess.call(['anipose', 'project-2d'])           
            
        if anipose_args['label_videos']:
            subprocess.call(['anipose', 'label-2d-proj']) 
            subprocess.call(['anipose', 'label-2d-filter']) 
            
        folders_with_new_info = ['pose-2d-proj',
                                 'pose-3d',
                                 'videos-labeled-filtered',
                                 'videos-2d-proj']    
                
    # move all new files back to primary anipose folder
    for folder_name in folders_with_new_info:
        src_dir = pjoin(task_path, folder_name)
        if os.path.isdir(src_dir):
            dst_dir = pjoin(anipose_args['date_dir'], folder_name)
            os.makedirs(dst_dir, exist_ok=True)
            src_files = glob.glob(pjoin(src_dir, '*'))
            print('moving files from \n%s to \n%s' % (src_dir, dst_dir))
            for f in src_files:
                shutil.move(f, dst_dir)
    if anipose_args['task_id'] == 0:
        shutil.move(pjoin(task_path, 'scorer_info.txt'), anipose_args['date_dir'])
            
    if anipose_args['task_id'] == 0:
        os.removedirs(pjoin(anipose_args['date_dir'], 'temp_anipose_processing'))

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
        args = {'dlc_path'     : '/project/nicho/projects/marmosets/dlc_project_files/simple_marmoset_model-Dalton-2023-04-28',
                'anipose_path' : '/project/nicho/data/marmosets/kinematics_videos/moth/TYJL',
                'dlc_iter'     : 'None',
                'train_frac'   : 'None',
                'snap_idx'     : -1,
                'parameter_set': 'marmoset',
                'extra_vars'   : ['False', 'True'],
                'dates'        : ['2021_02_05'],
                'ncams'        : 2}

    print('\n\n Beginning anipose processing at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)


    print(type(args['dlc_iter']), args['dlc_iter'], args['dlc_iter'] is None)
    print(type(args['train_frac']), args['train_frac'], args['train_frac'] is None)
    print(type(args['extra_vars']), args['extra_vars'])
    
    if not debugging:
        task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        n_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    else:
        task_id = 0
        n_tasks = 1

    iteration  = convert_string_inputs_to_int_float_or_bool(args['dlc_iter'])
    train_frac = convert_string_inputs_to_int_float_or_bool(args['train_frac'])
    extra_vars = convert_string_inputs_to_int_float_or_bool(args['extra_vars'])

    epPattern        = re.compile('_e[0-9]{3}')    
    epPattern_backup = re.compile('_event[0-9]{3}')    
    for date in args['dates']:
        ddir = pjoin(args['anipose_path'], date)
        videos = sorted(glob.glob(pjoin(ddir, 'avi_videos', '*.avi')))
        # event_nums = [int(re.findall(epPattern, os.path.basename(vid))[0].split('e')[-1]) for vid in videos]
        # events = np.unique(event_nums)
        # cam1_videos = [vidpath for vidpath in videos if 'cam1' in vidpath]
        # event_video_sizes = [(event, round(os.stat(vidpath).st_size/(1024**2))) for event, vidpath in zip(events, cam1_videos)]
        
        # full_vid_size_list = []
        # for event in event_nums:
        #     vid_size = [size_tuple[1] for size_tuple in event_video_sizes if size_tuple[0] == event][0]
        #     full_vid_size_list.append(vid_size)
        
        try:
            events = np.unique([int(re.findall(epPattern, os.path.basename(vid))[0].split('_e')[-1]) for vid in videos])
        except:
            events = np.unique([int(re.findall(epPattern_backup, os.path.basename(vid))[0].split('_event')[-1]) for vid in videos])
            
        cam1_videos = [vidpath for vidpath in videos if 'cam1' in vidpath]
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
            task_videos = [vid for vid in videos if int(re.findall(epPattern, os.path.basename(vid))[0].split('_e')[-1]) in task_events]   
        except:
            task_videos = [vid for vid in videos if int(re.findall(epPattern_backup, os.path.basename(vid))[0].split('_event')[-1]) in task_events]   


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
                        'ncams'           : args['ncams']}
    
        for key, val in anipose_args.items():
            print(key, ' : ', val, flush=True)
        
        if task_id == 0:
            for folder_name in ['pose-2d-unfiltered', 'pose-2d-viterbi', 
                                'pose-2d-viterbi_and_autoencoder', 'pose-3d', 
                                'videos-labeled-filtered', 'pose-2d-proj', 
                                'videos-2d-proj']:
                os.makedirs(pjoin(anipose_args['date_dir'], folder_name), exist_ok=True)
    
        compute_pose_with_anipose(anipose_args)
                                