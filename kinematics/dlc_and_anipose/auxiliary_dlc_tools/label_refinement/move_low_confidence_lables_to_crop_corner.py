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

def move_labels(args):  

    res_dir = Path(args['results_dir'])
    os.makedirs(res_dir / 'pose-2d-for-corrections', exist_ok=True)     

    h5_files = list((res_dir / args['corrections_input_dir']).glob('*.h5'))
    for f in h5_files:        
        pose   = pd.read_hdf(f)

        bp_level     = [level for level, name in enumerate(pose.columns.names) if name == 'bodyparts'][0]    
        coords_level = [level for level, name in enumerate(pose.columns.names) if name == 'coords'][0]  
        for label in pose.columns.get_level_values(bp_level).unique():
            partial_df = pose.loc[:, pose.columns.get_level_values(bp_level) == label]
            coords_columns = [col for col in partial_df.columns if col[coords_level] in ['x', 'y']]
            
            likelihoods    = partial_df.loc[:, partial_df.columns.get_level_values(coords_level) == 'likelihood'] 
            low_like_index = likelihoods.index[likelihoods.values.flatten() < 0.75]
            pose.loc[low_like_index, coords_columns] = np.full((low_like_index.size, 2), 0)
            
        pose.to_hdf(res_dir / 'pose-2d-for-corrections' / f.name, 
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

    args = {'results_dir'          : '/project/nicho/data/marmosets/kinematics_videos/moth/JLTY/2023_08_05/',
            'corrections_input_dir': 'pose-2d-viterbi_and_autoencoder',}
    
    move_labels(args)
                                
