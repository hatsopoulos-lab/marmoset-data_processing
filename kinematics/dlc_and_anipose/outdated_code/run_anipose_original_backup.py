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

iteration = None # 1
train_frac = None # 0.85
snapIdx = -1 #None # 7
dates = ['2021_02_11'] #'all' #['2021_03_29', '2021_03_30'] #'all' # ['2019_04_14', '2019_04_15']
add_events_only = False
only_3D = False

projectpath = '/home/marms/Documents/dlc_local/simple_joints_model-Dalton-2021-04-08'
aniposepath = '/media/marms/DATA/simple_joints_model_anipose'
# projectpath = '/home/marms/Documents/dlc_local/moth-tracking-jeff-2021-08-20'
# aniposepath = '/media/marms/DATA/prey_tracking_moths'

# projectpath = '/home/marms/Documents/dlc_local/bci-AshvinKumar-2022-01-09'
# aniposepath = '/media/marms/DATA/bci_ashvin_labeled'

param_category = ['filter', 'filter', 'triangulation', 'triangulation', 'triangulation', 'triangulation']
# test_params     = {'offset_threshold'       : 20, 
#                     'n_back'                 : 5, 
#                     'scale_smooth'           : 2, 
#                     'scale_length'           : 4, 
#                     'reproj_error_threshold' : 8,
#                     'score_threshold'        : 0.15} 
test_params     = {'offset_threshold'       : 20, 
                    'n_back'                 : 5, 
                    'scale_smooth'           : 4, 
                    'scale_length'           : 6, 
                    'reproj_error_threshold' : 8,
                    'score_threshold'        : 0.3} 

param_names  = list(test_params)
param_values = list(test_params.values())

os.chdir(aniposepath)
if dates == 'all':
    dates = glob.glob(os.path.join(aniposepath, '*'))
    dates = [os.path.basename(datepath) for datepath in dates if 'config' not in datepath]
    
dlc_config=os.path.join(projectpath,'config.yaml')
dlc_cfg=deeplabcut.auxiliaryfunctions.read_config(dlc_config)

original_iteration=dlc_cfg['iteration']
original_snapshotindex = dlc_cfg['snapshotindex']
original_train_fraction = dlc_cfg['TrainingFraction']

if iteration is not None:
    dlc_cfg['iteration'] = iteration
if snapIdx is not None:
    dlc_cfg['snapshotindex'] = snapIdx
if train_frac is not None:
    dlc_cfg['TrainingFraction'] = [train_frac]

deeplabcut.auxiliaryfunctions.write_config(dlc_config, dlc_cfg)

ani_config = os.path.join(aniposepath, 'config.toml')
ani_cfg = toml.load(ani_config)

for cat, key, param in zip(param_category, param_names, param_values):
    ani_cfg[cat][key] = param

if add_events_only:
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
if not only_3D:        
    subprocess.call(['anipose', 'analyze'])

ani_cfg['filter']['type'] = 'viterbi'

with open(ani_config, 'w') as f:
    toml.dump(ani_cfg, f)

if not only_3D:
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
        
        if add_events_only:
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

if add_events_only:
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

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--jpg_dir", required=True,
        help="path to temporary directory holding jpg files for task and marmoset pair. E.g. /scratch/midway3/daltonm/kinematics_jpgs/")
    args = vars(ap.parse_args())

                                