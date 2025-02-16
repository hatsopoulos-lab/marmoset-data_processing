#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 16:17:54 2022

@author: daltonm
"""

import pandas as pd
import glob
import os

dlc_scorer = 'AshvinKumar'
labels_path = '/project/nicho/projects/bci/dlc_project_files/bci-AshvinKumar-2022-01-09/labeled-data'

label_folders = sorted(glob.glob(os.path.join(labels_path, '*')))

for fold in label_folders:
    label_data = pd.read_hdf(os.path.join(fold, 'CollectedData_%s.h5' % dlc_scorer))
    existing_idx = label_data.index
    if type(existing_idx) == pd.core.indexes.base.Index:
        idx_array = [[], [], []]
        for idx in existing_idx:
            info = idx.split('/')
            
            idx_array[0].append(info[0])
            idx_array[1].append(info[1])
            idx_array[2].append(info[2])
        
        good_index = pd.MultiIndex.from_arrays(idx_array)
        label_data.index = good_index
        
        label_data.to_csv(os.path.join(fold, "CollectedData_%s.csv" % dlc_scorer))
        label_data.to_hdf(os.path.join(fold, "CollectedData_%s.h5" % dlc_scorer), "df_with_missing", mode="w")
            