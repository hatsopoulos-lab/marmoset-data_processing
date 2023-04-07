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
import glob
import shutil
import time

def compute_pose_with_anipose(anipose_args):  

    aniposepath = anipose_args['aniposepath']
    
    param_category = ['filter', 'filter', 'triangulation', 'triangulation', 'triangulation', 'triangulation']

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
                       'reproj_error_threshold' : 8,
                       'score_threshold'        : 0.3}
 
    param_names  = list(test_params)
    param_values = list(test_params.values())
    
    if anipose_args['task_id'] == 0:
        dlc_config=os.path.join(anipose_args['projectpath'],'config.yaml')
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
    ani_config = os.path.join(anipose_args['date_dir'], 'config.toml')
    ani_template = os.path.join(aniposepath, 'config.toml')
    shutil.copy(ani_template, ani_config)    
    
    os.chdir(anipose_args['date_dir'])
    
    ani_cfg = toml.load(ani_config) 
    for cat, key, param in zip(param_category, param_names, param_values):
        ani_cfg[cat][key] = param
    ani_cfg['model_folder'] = anipose_args['projectpath']
    ani_cfg['nesting'] = 0
    with open(ani_config, 'w') as f:
        toml.dump(ani_cfg, f)

    if anipose_args['task_id'] == 0:
        subprocess.call(['anipose', 'calibrate'])
    else:
        task_0_path = os.path.join(anipose_args['date_dir'], 'temp_anipose_processing', '0')
        while not os.path.isdir(task_0_path):
            time.sleep(5)
    
    # copy config file and alibration results temp_anipose_processing/TASKID folders
    task_path = os.path.join(anipose_args['date_dir'], 'temp_anipose_processing', str(anipose_args['task_id']))
    os.makedirs(os.path.join(task_path, 'calibration'), exist_ok=True)
    os.makedirs(os.path.join(task_path, 'avi_videos') , exist_ok=True)
    shutil.copy(ani_config, task_path)
    shutil.copy(os.path.join(anipose_args['date_dir'], 'calibration', 'calibration.toml'), os.path.join(task_path, 'calibration'))
    
    # move into directory for task processing
    os.chdir(task_path)
    tmp_ani_config_path = os.path.join(task_path, 'config.toml')
    tmp_ani_cfg = toml.load(tmp_ani_config_path) 
    # print('applying anipose to videos %s thru %s' % (os.path.basename(task_bright_videos[0]), 
    #                                                  os.path.basename(task_bright_videos[-1])))
    
    print('\n' + os.getcwd(), flush=True)
    if not anipose_args['only_3D']:        

        # copy the videos for this task into the TASK_ID folder
        for vidpath in anipose_args['task_videos']:
            shutil.copy(vidpath, os.path.join(task_path, 'avi_videos'))        

        tmp_ani_cfg['filter']['type'] = 'viterbi'
        tmp_ani_cfg['pipeline']['pose_2d_filter'] = 'pose-2d-viterbi'
        tmp_ani_cfg['pipeline']['pose_2d'] = 'pose-2d-unfiltered'
        with open(tmp_ani_config_path, 'w') as f:
            toml.dump(tmp_ani_cfg, f)

        subprocess.call(['anipose', 'analyze'])
        subprocess.call(['anipose', 'filter'])     
    
        tmp_ani_cfg['filter']['type'] = 'autoencoder'
        tmp_ani_cfg['pipeline']['pose_2d_filter'] = 'pose-2d-viterbi_and_autoencoder'
        tmp_ani_cfg['pipeline']['pose_2d'] = 'pose-2d-viterbi'
        with open(tmp_ani_config_path, 'w') as f:
            toml.dump(tmp_ani_cfg, f)
        
        # move files to date folder, train autoencoder, then copy autoencoder to task folders 
        ani_cfg['filter']['type'] = 'autoencoder'
        ani_cfg['pipeline']['pose_2d'] = 'pose-2d-viterbi'
        for src_file in sorted(glob.glob(os.path.join(task_path, 'pose-2d-viterbi', '*'))):
            dst_file = os.path.join(anipose_args['date_dir'], 'pose-2d-viterbi', os.path.basename(src_file))
            shutil.copy(src_file, dst_file)
        
        if anipose_args['task_id'] == 0:
            os.chdir(anipose_args['date_dir'])
            while len(glob.glob(os.path.join(anipose_args['date_dir'], 'pose-2d-viterbi'))) < len(glob.glob(os.path.join(anipose_args['date_dir'], 'avi_videos'))):
                time.sleep(10)
            subprocess.call(['anipose', 'train-autoencoder'])
            shutil.copy(os.path.join(anipose_args['date_dir'], 'autoencoder.pickle'), task_path)
        else:
            while not os.path.isfile(os.path.join(task_0_path, 'autoencoder.pickle')):
                time.sleep(10)
            shutil.copy(os.path.join(anipose_args['date_dir'], 'autoencoder.pickle'), task_path)
        
        os.chdir(task_path)
        subprocess.call(['anipose', 'filter'])
        
        folders_with_new_info = ['pose-2d-viterbi_and_autoencoder', 
                                 'pose-2d-viterbi', 
                                 'pose-2d-unfiltered',
                                 'pose-2d-proj',
                                 'pose-3d',
                                 'videos-labeled-filtered',
                                 'videos-labeled-proj']
                
    else:
        for vidpath in anipose_args['task_videos']:
            base_path, filename = os.path.split(vidpath)
            base_path = os.path.dirname(base_path)
            filename = os.path.splitext(filename)[0] + '.h5'
            pose_path = os.path.join(base_path, 'pose-2d-viterbi_and_autoencoder', filename)
            os.makedirs(os.path.join(task_path, 'pose-2d-viterbi_and_autoencoder'))
            shutil.copy(pose_path, os.path.join(task_path, 'pose-2d-viterbi_and_autoencoder')) 
        
        tmp_ani_cfg['pipeline']['pose_2d_filtered'] = 'pose-2d-viterbi_and_autoencoder'
        tmp_ani_cfg['pipeline']['pose_2d'] = 'pose-2d-viterbi'
        with open(tmp_ani_config_path, 'w') as f:
            toml.dump(tmp_ani_cfg, f)
            
        folders_with_new_info = ['pose-2d-proj',
                                 'pose-3d',
                                 'videos-labeled-filtered',
                                 'videos-labeled-proj']
    
    subprocess.call(['anipose', 'triangulate']) 
    subprocess.call(['anipose', 'project-2d'])     

    if anipose_args['label_videos']:
        subprocess.call(['anipose', 'label-2d-proj']) 
        subprocess.call(['anipose', 'label-2d-filter']) 
                
    # move all new files back to primary anipose folder
    for folder_name in folders_with_new_info:
        src_dir = os.path.join(task_path, folder_name)
        if os.path.isdir(src_dir):
            dst_dir = os.path.join(anipose_args['date_dir'], folder_name)
            os.makedirs(dst_dir, exist_ok=True)
            src_files = glob.glob(os.path.join(src_dir, '*'))
            for f in src_files:
                shutil.move(f, dst_dir)
    if anipose_args['task_id'] == 0:
        shutil.move(os.path.join(task_path, 'scorer_info.txt'), anipose_args['date_dir'])
    
    ###### NEED TO REMOVE TEMP FOLDERS HERE #####
    
    os.remove(os.path.join(aniposepath, 'autoencoder.pickle'))

    
    if anipose_args['task_id'] == 0:
        print("resetting snapshotindex and iteration")
        dlc_cfg['iteration'] = original_iteration
        dlc_cfg['snapshotindex'] = original_snapshotindex
        dlc_cfg['TrainingFraction'] = original_train_fraction
        deeplabcut.auxiliaryfunctions.write_config(dlc_config, dlc_cfg)
    
    # print('resetting anipose pipeline variables in %s' % ani_config)
    # ani_cfg['pipeline']['pose_2d_filter'] = 'pose-2d-viterbi'
    # ani_cfg['pipeline']['pose_2d'] = 'pose-2d-unfiltered'
    # with open(ani_config, 'w') as f:
    #     toml.dump(ani_cfg, f)

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
    args = vars(ap.parse_args())

    print('\n\n Beginning anipose processing at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)


    print(type(args['dlc_iter']), args['dlc_iter'], args['dlc_iter'] is None)
    print(type(args['train_frac']), args['train_frac'], args['train_frac'] is None)
    print(type(args['extra_vars']), args['extra_vars'])
    
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    n_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))

    iteration  = convert_string_inputs_to_int_float_or_bool(args['dlc_iter'])
    train_frac = convert_string_inputs_to_int_float_or_bool(args['train_frac'])
    extra_vars = convert_string_inputs_to_int_float_or_bool(args['extra_vars'])

    videos = []
    for date in args['dates']:
        ddir = os.path.join(args['anipose_path'], date)
        videos.extend(sorted(glob.glob(os.path.join(ddir, 'avi_videos', '*.avi'))))    
        
        task_idx_cutoffs = np.floor(np.linspace(0, len(videos), n_tasks+1))   
        task_idx_cutoffs = [int(cut) for cut in task_idx_cutoffs]
        task_videos      = videos[task_idx_cutoffs[task_id] : task_idx_cutoffs[task_id+1]]    

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
                        'date_dir'        : ddir}
    
        for key, val in anipose_args.items():
            print(key, ' : ', val, flush=True)
        
        if task_id == 0:
            for folder_name in ['pose-2d-unfiltered', 'pose-2d-viterbi', 
                                'pose-2d-viterbi_and_autoencoder', 'pose-3d', 
                                'videos-labeled-filtered', 'pose-2d-proj', 
                                'videos-labeled-proj']:
                os.makedirs(os.path.join(anipose_args['date_dir'], folder_name), exist_ok=True)
    
        compute_pose_with_anipose(anipose_args)
                                