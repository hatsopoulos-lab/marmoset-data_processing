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

def merge_labels(args):  

    res_dir = Path(args['results_dir'])
    
    orig_path = res_dir / args['corrections_input_dir']
    corr_path = res_dir / 'pose-2d-for-corrections'
    merge_path = res_dir / 'pose-2d-merged-corrections'    
    os.makedirs(merge_path, exist_ok=True)     

    cam_pattern = re.compile('cam[0-9]{1}')
    event_pattern = re.compile('_e[0-9]{3}') 

    h5_files = list((orig_path).glob('*.h5'))
    if h5_files is not None and args['corrected_events'] is not None:
        h5_files     = [f for f in h5_files if int(re.findall(event_pattern, f.stem)[0].split('_e')[-1]) in args['corrected_events']]   

    for f in h5_files:        
        f_corr = corr_path / f.name
        if not f_corr.is_file():
            continue
        
        cam = re.findall(cam_pattern, f.stem)[0]
        
        pose_orig = pd.read_hdf(f)
        pose_corr = pd.read_hdf(f_corr)

        bp_level     = [level for level, name in enumerate(pose_orig.columns.names) if name == 'bodyparts'][0]    
        coords_level = [level for level, name in enumerate(pose_orig.columns.names) if name == 'coords'][0]  
        for label in pose_orig.columns.get_level_values(bp_level).unique():
            partial_orig = pose_orig.loc[:, pose_orig.columns.get_level_values(bp_level) == label]
            partial_corr = pose_corr.loc[:, pose_corr.columns.get_level_values(bp_level) == label]
            
            coords_columns = [col for col in partial_orig.columns if col[coords_level] in ['x', 'y']]
            like_column    = [col for col in partial_orig.columns if col[coords_level] == 'likelihood'][0]
            
            if (label in args['one_side_labels']['right'] and cam in args['left_cams']) or (label in args['one_side_labels']['left']  and cam in args['right_cams']):
                pose_orig.loc[:, like_column] = np.full_like(pose_orig.loc[:, like_column], 0)
            
            distance = np.sqrt(np.sum(np.square(partial_corr.loc[:, coords_columns].values - partial_orig.loc[:, coords_columns]), axis=1))
            corrected = np.where(distance > 0)[0]
            pdIdx = pd.IndexSlice
            visible   = np.where((partial_corr.loc[:, pdIdx[:, :, 'x']].values.flatten() > 50) & (partial_corr.loc[:, pdIdx[:, :, 'y']].values.flatten() > 50))[0]
            altered_index = partial_orig.index[np.intersect1d(corrected, visible)]
            pose_orig.loc[altered_index, coords_columns] = pose_corr.loc[altered_index, coords_columns] 
            pose_orig.loc[altered_index, like_column]    = np.full((altered_index.size,), 1)      
            
        pose_orig.to_hdf(merge_path / f.name, 
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

    args = {'results_dir'          : '/project/nicho/data/marmosets/kinematics_videos/moth/JLTY/2023_08_05',
            'corrections_input_dir': 'pose-2d-viterbi_and_autoencoder',
            'corrected_events'     : None,
            'left_cams'            : ['cam2', 'cam4'],
            'right_cams'           : ['cam1', 'cam3'],
            'one_side_labels'      : dict(right=['r-head-corner', 
                                                 'r-head-under-tab', 'r-shoulder',
                                                 'r-elbow', 'x', 'partition-top-right'],
                                          left=['l-head-corner', 
                                                'l-head-under-tab', 'l-shoulder', 
                                                'l-elbow', 'origin', 'partition-top-left']),
            }
    
    merge_labels(args)
                                
