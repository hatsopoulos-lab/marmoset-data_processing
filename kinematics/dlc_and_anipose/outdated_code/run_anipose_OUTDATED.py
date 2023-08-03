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
    
    os.chdir(aniposepath)
        
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
    ani_cfg['model_folder'] = anipose_args['projectpath']
    with open(ani_config, 'w') as f:
        toml.dump(ani_cfg, f)

    subprocess.call(['anipose', 'calibrate'])
    
    print(os.getcwd(), flush=True)
    if not anipose_args['only_3D']:        
 
        ani_cfg['filter']['type'] = 'viterbi'
        ani_cfg['pipeline']['pose_2d_filter'] = 'pose-2d-viterbi'
        ani_cfg['pipeline']['pose_2d'] = 'pose-2d-unfiltered'
        with open(ani_config, 'w') as f:
            toml.dump(ani_cfg, f)

        subprocess.call(['anipose', 'analyze'])
        subprocess.call(['anipose', 'filter'])     
    
        ani_cfg['filter']['type'] = 'autoencoder'
        ani_cfg['pipeline']['pose_2d_filter'] = 'pose-2d-viterbi_and_autoencoder'
        ani_cfg['pipeline']['pose_2d'] = 'pose-2d-viterbi'
        with open(ani_config, 'w') as f:
            toml.dump(ani_cfg, f)
            
        subprocess.call(['anipose', 'train-autoencoder'])
        subprocess.call(['anipose', 'filter'])
    else:
        ani_cfg['pipeline']['pose_2d_filtered'] = 'pose-2d-viterbi_and_autoencoder'
        ani_cfg['pipeline']['pose_2d'] = 'pose-2d-viterbi'
        with open(ani_config, 'w') as f:
            toml.dump(ani_cfg, f)   
    
    subprocess.call(['anipose', 'triangulate']) 
    
    if anipose_args['label_videos']:
        subprocess.call(['anipose', 'project-2d']) 
        # subprocess.call(['anipose', 'label-2d-proj']) 
        subprocess.call(['anipose', 'label-2d-filter']) 
        # subprocess.call(['anipose', 'label-3d']) 
        # subprocess.call(['anipose', 'label-combined']) 
                
    os.remove(os.path.join(aniposepath, 'autoencoder.pickle'))
    
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
    args = vars(ap.parse_args())

    print(type(args['dlc_iter']), args['dlc_iter'], args['dlc_iter'] is None)
    print(type(args['train_frac']), args['train_frac'], args['train_frac'] is None)
    print(type(args['extra_vars']), args['extra_vars'])

    iteration  = convert_string_inputs_to_int_float_or_bool(args['dlc_iter'])
    train_frac = convert_string_inputs_to_int_float_or_bool(args['train_frac'])
    extra_vars = convert_string_inputs_to_int_float_or_bool(args['extra_vars'])

    anipose_args = {'projectpath'     : args['dlc_path'],
                    'aniposepath'     : args['anipose_path'],
                    'iteration'       : iteration,
                    'train_frac'      : train_frac,
                    'snap_idx'        : args['snap_idx'],
                    'parameter_set'   : args['parameter_set'],
                    'only_3D'         : extra_vars[0],
                    'label_videos'    : extra_vars[1]}
    
    print(anipose_args, flush=True)

    compute_pose_with_anipose(anipose_args)
                                