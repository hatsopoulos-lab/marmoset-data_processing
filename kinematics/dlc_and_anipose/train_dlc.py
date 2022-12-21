# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:21:33 2021

@author: Dalton
"""

import deeplabcut
import argparse
import os
import glob

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dlc_path", required=True, type=str,
        help="path to dlc project. E.g. '/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26'")
    ap.add_argument("-i", "--init_weights", required=True, type=str,
        help="path to init_weights. E.g. '/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26/dlc-models/iteration-2/full_marmoset_modelJul26-trainset95shuffle1/train/snapshot-350000'")
    ap.add_argument("-o", "--overwrite", required=True, type=str,
        help="True/False whether overwriting training dataset is allowed")
    ap.add_argument("-m", "--maxiters", required=True, type=int,
     	help="number of iterations to stop training")
    args = vars(ap.parse_args())
    
    shuffle = 1
    trainsetindex = 0
    
    dlc_config = os.path.join(args['dlc_path'], 'config.yaml')
    dlc_cfg = deeplabcut.auxiliaryfunctions.read_config(dlc_config)
    trainset_name = dlc_cfg['Task'] + dlc_cfg['date'] + '-trainset' + str(int(dlc_cfg['TrainingFraction'][trainsetindex] * 100)) + 'shuffle' + str(shuffle) 
    
    print('creating training dataset')
    trainset_exists = len(glob.glob(os.path.join(args['dlc_path'], 'training-datasets', 'iteration-' + str(dlc_cfg['iteration']), '*'))) > 0
    if args['overwrite'] == 'False': 
        if not trainset_exists:
            deeplabcut.create_training_dataset(dlc_config)
        else:
            print('training set already exists')
    else:
        deeplabcut.create_training_dataset(dlc_config)
    
    train_config = os.path.join(args['dlc_path'], 
                                'dlc-models', 
                                'iteration-'+str(dlc_cfg['iteration']),
                                trainset_name,
                                'train',
                                'pose_cfg.yaml')
    
    train_cfg=deeplabcut.auxiliaryfunctions.read_plainconfig(train_config)
    if args['init_weights'] != 'None':
        print('changing init_weights from %s to %s' % (train_cfg['init_weights'], args['init_weights']))
        train_cfg['init_weights'] = args['init_weights']
        deeplabcut.auxiliaryfunctions.write_plainconfig(train_config, train_cfg)
    
    print('beginning to train network \n\n', flush=True)
    deeplabcut.train_network(dlc_config, 
                             maxiters = args['maxiters'], 
                             max_snapshots_to_keep = 5,
                             displayiters=1000, 
                             saveiters=20000, 
                             gputouse=None)
