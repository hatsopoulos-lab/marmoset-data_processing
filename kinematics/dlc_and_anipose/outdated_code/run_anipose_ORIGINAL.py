# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:21:33 2021

@author: Dalton
"""

import shutil
import subprocess
import os
import deeplabcut
import glob
import toml
import argparse

def compute_pose_with_anipose(anipose_args):

    aniposepath = anipose_args['aniposepath']

    param_category = ['filter', 'filter', 'triangulation', 'triangulation', 'triangulation', 'triangulation']

    if 'marm' in anipose_args['parameter_set'].lower():
        test_params = {'offset_threshold'       : 20, 
                       'n_back'                 : 5, 
                       'scale_smooth'           : 4, 
                       'scale_length'           : 6, 
                       'reproj_error_threshold' : 8,
                       'score_threshold'        : 0.3} 
    elif 'bci' in anipose_args['parameter_set'].lower():
        test_params = {'offset_threshold'       : 20, 
                       'n_back'                 : 5, 
                       'scale_smooth'           : 4, 
                       'scale_length'           : 6, 
                       'reproj_error_threshold' : 8,
                       'score_threshold'        : 0.3}
        
    # test_params     = {'offset_threshold'       : 20, 
    #                     'n_back'                 : 5, 
    #                     'scale_smooth'           : 2, 
    #                     'scale_length'           : 4, 
    #                     'reproj_error_threshold' : 8,
    #                     'score_threshold'        : 0.15} 

    
    param_names  = list(test_params)
    param_values = list(test_params.values())
    
    os.chdir(aniposepath)
    if anipose_args['dates'][0] == 'all':
        dates = glob.glob(os.path.join(aniposepath, '*'))
        dates = [os.path.basename(datepath) for datepath in dates if 'config' not in datepath]
    else:
        dates = anipose_args['dates']
        
    dlc_config=os.path.join(anipose_args['projectpath'],'config.yaml')
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
    
    ani_config = os.path.join(aniposepath, 'config.toml')
    ani_cfg = toml.load(ani_config)
    
    for cat, key, param in zip(param_category, param_names, param_values):
        ani_cfg[cat][key] = param
    
    if anipose_args['add_events_only']:
        for date in dates:
            shutil.move(os.path.join(aniposepath, date, 'pose-2d'),
                        os.path.join(aniposepath, date, 'pose-2d_tmp'))
            shutil.move(os.path.join(aniposepath, date, 'pose-2d-viterbi-only'),
                        os.path.join(aniposepath, date, 'pose-2d-viterbi-only_tmp'))
            shutil.move(os.path.join(aniposepath, date, 'pose-2d-raw'),
                        os.path.join(aniposepath, date, 'pose-2d-raw_tmp'))
            shutil.move(os.path.join(aniposepath, date, 'pose-2d-filtered'),
                        os.path.join(aniposepath, date, 'pose-2d-filtered_tmp'))
            
            shutil.copytree(os.path.join(aniposepath, date, 'pose-2d-raw_tmp'), 
                            os.path.join(aniposepath, date, 'pose-2d'))
            shutil.copytree(os.path.join(aniposepath, date, 'pose-2d-viterbi-only_tmp'), 
                            os.path.join(aniposepath, date, 'pose-2d-filtered'))

    subprocess.call(['anipose', 'calibrate'])
    
    print(os.getcwd(), flush=True)
    if not anipose_args['only_3D']:        
        subprocess.call(['anipose', 'analyze'])
    
    ani_cfg['filter']['type'] = 'viterbi'
    
    with open(ani_config, 'w') as f:
        toml.dump(ani_cfg, f)
    
    if not anipose_args['only_3D']:
        subprocess.call(['anipose', 'filter'])
    
        for date in dates:
            shutil.copytree(os.path.join(aniposepath, date, 'pose-2d'), 
                            os.path.join(aniposepath, date, 'pose-2d-raw'))
            
            shutil.copytree(os.path.join(aniposepath, date, 'pose-2d-filtered'), 
                            os.path.join(aniposepath, date, 'pose-2d-viterbi-only'))
                
            shutil.rmtree(os.path.join(aniposepath, date, 'pose-2d'))
            shutil.rmtree(os.path.join(aniposepath, date, 'pose-2d-filtered'))
            
            shutil.copytree(os.path.join(aniposepath, date, 'pose-2d-viterbi-only'),
                            os.path.join(aniposepath, date, 'pose-2d'))
            
            if anipose_args['add_events_only']:
                shutil.copytree(os.path.join(aniposepath, date, 'pose-2d-filtered_tmp'), 
                                os.path.join(aniposepath, date, 'pose-2d-filtered'))        
    
        ani_cfg['filter']['type'] = 'autoencoder'
    
        with open(ani_config, 'w') as f:
            toml.dump(ani_cfg, f)
            
        subprocess.call(['anipose', 'train-autoencoder'])
        subprocess.call(['anipose', 'filter'])
    
    subprocess.call(['anipose', 'triangulate']) 
    subprocess.call(['anipose', 'label-2d-filter']) 
    subprocess.call(['anipose', 'label-3d']) 
    subprocess.call(['anipose', 'label-combined']) 
    
    if anipose_args['add_events_only']:
        for date in dates:
            tmp_folders = glob.glob(os.path.join(aniposepath, date, '*_tmp'))
            for folder in tmp_folders:
                shutil.rmtree(folder)
                
    os.remove(os.path.join(aniposepath, 'autoencoder.pickle'))
    
    print("resetting snapshotindex and iteration")
    dlc_cfg['iteration'] = original_iteration
    dlc_cfg['snapshotindex'] = original_snapshotindex
    dlc_cfg['TrainingFraction'] = original_train_fraction
    deeplabcut.auxiliaryfunctions.write_config(dlc_config, dlc_cfg)

def convert_boolean_vars(orig_var):
    if type(orig_var) == str:
        orig_var = [orig_var]
    
    converted_var = []
    for v in orig_var:
        if v.lower() == 'false':
            converted_var.append(False)
        elif v.lower() == 'true':
            converted_var.append(True)
        elif v.lower() == 'none':
            converted_var.append(None)
        else:
            converted_var.append(v)
    
    if len(converted_var) == 1:
        converted_var = converted_var[0]
            
    return converted_var

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dlc_path", required=True,
        help="path to dlc project. E.g. '/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26'")
    ap.add_argument("-a", "--anipose_path", required=True,
        help="path to anipose project. E.g. '/project/nicho/data/marmosets/test'")
    ap.add_argument("-i", "--dlc_iter", required=True,
        help="dlc project iteration to use. If 'None', use the iteration already in config.yaml")
    ap.add_argument("-f", "--train_frac", required=True,
        help="dlc training fraction to use. If 'None', use the training fraction already in config.yaml")
    ap.add_argument("-s", "--snap_idx", required=True,
        help="dlc project snapshot index to use. '-1' will use the last saved snapshot")
    ap.add_argument("-p", "--parameter_set", required=True,
        help="parameter set to use. E.g. 'marms' or 'bci'")
    ap.add_argument("-dt", "--dates", required=True, nargs='+',
        help="dates to analyze. Input is a list of dates such as '2022_05_17 2022_04_13' or can be 'all' to analyze all dates")
    ap.add_argument("-v", "--extra_vars", required=True, nargs='+',
        help="Additional variables to modify functionality: [add_events_only, only3D]. Input is a list of True/False, e.g. 'False False'")
    args = vars(ap.parse_args())

    iteration  = convert_boolean_vars(args['dlc_iter'])
    train_frac = convert_boolean_vars(args['train_frac'])
    extra_vars = convert_boolean_vars(args['extra_vars'])
    
    # if len(args['dates']) == 1:
    #     args['dates'] = args['dates'][0]

    anipose_args = {'projectpath'     : args['dlc_path'],
                    'aniposepath'     : args['anipose_path'],
                    'iteration'       : iteration,
                    'train_frac'      : train_frac,
                    'snap_idx'        : int(args['snap_idx']),
                    'dates'           : args['dates'],
                    'parameter_set'   : args['parameter_set'],
                    'add_events_only' : extra_vars[0],
                    'only_3D'         : extra_vars[1]}
    
    print(anipose_args, flush=True)

    compute_pose_with_anipose(anipose_args)
                                