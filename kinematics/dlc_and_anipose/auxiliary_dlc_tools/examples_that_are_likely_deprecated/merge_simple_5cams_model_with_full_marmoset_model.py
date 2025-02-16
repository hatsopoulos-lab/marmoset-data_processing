# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:49:39 2022

@author: Dalton
"""

import os
import glob
import pandas as pd
import numpy as np

base = 'T:/projects/marmosets/dlc_project_files/'

new_path = os.path.join(base, 'full_marmoset_model-Dalton-2022-07-26/labeled-data/')
old_path = os.path.join(base, 'simple_joints_model-Dalton-2021-04-08/labeled-data/')

old_label_folders = sorted(glob.glob(os.path.join(old_path, '*')))
new_label_folders = sorted(glob.glob(os.path.join(new_path, '*')))

old_idxs = [0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17]
new_idxs = [26, 27, 32, 33, 34, 35, 36, 37, 28, 29, 30, 31, 52, 53, 54, 55, 56, 57]

template_data = pd.read_hdf(glob.glob(os.path.join(new_label_folders[0], 'CollectedData*.h5'))[0])
template_data.iloc[:, :] = np.nan

for fold in old_label_folders:
    old_label_path = glob.glob(os.path.join(fold, 'CollectedData*.h5'))
    try:
        old_data = pd.read_hdf(old_label_path[0])
        new_data_path = [os.path.join(new_fold, os.path.basename(old_label_path[0])) for new_fold in new_label_folders 
                         if os.path.basename(new_fold) == os.path.basename(fold)][0]
        new_data = template_data
        new_data.index = old_data.index
        new_data.iloc[:, new_idxs] = old_data.iloc[:, old_idxs]
        
        new_data_path, _ = os.path.splitext(new_data_path)     
        print(new_data_path)               
        new_data.to_csv(new_data_path + '.csv')
        new_data.to_hdf(new_data_path + '.h5' , "df_with_missing")
    except:
        continue