# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:21:33 2021

@author: Dalton
"""

import deeplabcut
import argparse
import shutil
from pathlib import Path

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dlc_path", required=True, type=str,
        help="path to dlc project. E.g. '/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26'")
    ap.add_argument("-t", "--template_path", required=True, type=str,
        help="path to template pytorch_config.yaml file. E.g. '/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26/pytorch_config_template.yaml'")
    ap.add_argument("-i", "--init_weights", required=True, type=str,
        help="path to init_weights. E.g. '/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26/dlc-models/iteration-2/full_marmoset_modelJul26-trainset95shuffle1/train/snapshot-350000'")
    ap.add_argument("-o", "--overwrite", required=True, type=str,
        help="True/False whether overwriting training dataset is allowed")
    ap.add_argument("-m", "--maxiters", required=True, type=int,
     	help="number of iterations to stop training")
    ap.add_argument("-b", "--batch_size", required=True, type=str,
     	help="batch size (higher will improve generalization but above 8 may cause out of memory error")
    args = vars(ap.parse_args())
    
    shuffle = 1
    trainsetindex = 0
    
    dlc_path = Path(args['dlc_path'])
    dlc_config = dlc_path / 'config.yaml'
    dlc_cfg = deeplabcut.auxiliaryfunctions.read_config(dlc_config)
    # trainset_name = dlc_cfg['Task'] + dlc_cfg['date'] + '-trainset' + str(int(dlc_cfg['TrainingFraction'][trainsetindex] * 100)) + 'shuffle' + str(shuffle) 
    trainset_name = f"{dlc_cfg['Task']}{dlc_cfg['date']}-trainset{str(int(dlc_cfg['TrainingFraction'][trainsetindex]*100))}shuffle{str(shuffle)}" 

    print('creating training dataset')
    trainset_exists = len(list((dlc_path / 'training-datasets' / f'iteration-{dlc_cfg["iteration"]}').glob('*'))) > 0
    # trainset_exists = len(glob.glob(os.path.join(args['dlc_path'], 'training-datasets', 'iteration-' + str(dlc_cfg['iteration']), '*'))) > 0
    if args['overwrite'] == 'False': 
        if not trainset_exists:
            deeplabcut.create_training_dataset(dlc_config)
        else:
            print('training set already exists')
    else:
        deeplabcut.create_training_dataset(dlc_config)
    
    
    train_path = dlc_path / 'dlc-models-pytorch' / f'iteration-{dlc_cfg["iteration"]}' / trainset_name / 'train'
    train_config   = train_path / 'pose_cfg.yaml'
    pytorch_config = train_path / 'pytorch_config.yaml'
    
    shutil.copy(pytorch_config, train_path / 'pytorch_cf_orig.yaml')
    pytorch_config.unlink()
    shutil.copy(args['template_path'], pytorch_config)
    pytorch_cfg = deeplabcut.auxiliaryfunctions.read_plainconfig(pytorch_config)
    pytorch_cfg['metadata']['pose_config_path'] = str(train_config)
    
    train_cfg=deeplabcut.auxiliaryfunctions.read_plainconfig(train_config)
    if args['init_weights'] != 'None':
        print('changing init_weights from %s to %s' % (train_cfg['init_weights'], args['init_weights']))
        train_cfg['init_weights'] = args['init_weights']
    if args['batch_size'] != 'None':
        print('changing batch_size from %s to %s' % (train_cfg['batch_size'], args['batch_size']))
        train_cfg['batch_size'] = int(args['batch_size'])
        pytorch_cfg['train_settings']['batch_size'] = int(args['batch_size'])    
        
    deeplabcut.auxiliaryfunctions.write_plainconfig(train_config, train_cfg)    
    deeplabcut.auxiliaryfunctions.write_plainconfig(pytorch_config, pytorch_cfg)    
    
    print('beginning to train network \n\n', flush=True)
    deeplabcut.train_network(dlc_config)
    
    #####################################
    # changed_config = False
    # train_cfg=deeplabcut.auxiliaryfunctions.read_plainconfig(train_config)
    # if args['init_weights'] != 'None':
    #     print('changing init_weights from %s to %s' % (train_cfg['init_weights'], args['init_weights']))
    #     train_cfg['init_weights'] = args['init_weights']
    #     changed_config = True
    # if args['batch_size'] != 'None':
    #     print('changing batch_size from %s to %s' % (train_cfg['batch_size'], args['batch_size']))
    #     train_cfg['batch_size'] = int(args['batch_size'])
    #     changed_config = True
        
    # if changed_config:
    #     deeplabcut.auxiliaryfunctions.write_plainconfig(train_config, train_cfg)    
    
    # print('beginning to train network \n\n', flush=True)
    # deeplabcut.train_network(dlc_config, 
    #                           maxiters = args['maxiters'], 
    #                           max_snapshots_to_keep = 5,
    #                           displayiters=1000, 
    #                           saveiters=20000, 
    #                           gputouse=None)
