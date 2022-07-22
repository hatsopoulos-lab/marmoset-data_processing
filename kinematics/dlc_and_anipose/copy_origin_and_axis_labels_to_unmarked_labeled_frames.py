# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:58:36 2022

@author: Dalton
"""

import pandas as pd
import numpy as np
import glob
import os

nCams = 2
labels_path = '/home/marms/Documents/dlc_local/simple_joints_model-Dalton-2021-04-08/labeled-data'

fix_bumped_apparatus_origin_points = True
if fix_bumped_apparatus_origin_points:
    fix_date = '2021_02_11'
    start_event = 61 
    end_event = None # put None if end event is the last event

label_folders = glob.glob(os.path.join(labels_path, '*'))

dates = []
for f in label_folders:
    dates.append(f.split('TYJL_')[1][:10])

unique_dates = []
for date in dates:
    if date not in unique_dates:
        unique_dates.append(date)

if fix_bumped_apparatus_origin_points:
    unique_dates = [fix_date]
        
for date in unique_dates:
    for cam in range(1, nCams+1):
        cam_folders = sorted(glob.glob(os.path.join(labels_path, '*%s*cam%d*' % (date, cam))))
        if fix_bumped_apparatus_origin_points:
            if end_event is None:
                end_event = int(cam_folders[-1].split('event')[1][:3])
            cam_folders = [cFold for cFold in cam_folders 
                           if int(cFold.split('event')[1][:3]) >= start_event
                           and int(cFold.split('event')[1][:3]) <= end_event]
        first_data = pd.read_hdf(os.path.join(cam_folders[0], 'CollectedData_Dalton.h5'))
        axes_label_frame = np.where(~np.isnan(first_data.loc[:, ('Dalton', 'origin', 'x')]))[0][0]
        origin_label_column = np.where(first_data.columns == ('Dalton', 'origin', 'x'))[0][0]        
        axes_labels = np.array(first_data.iloc[axes_label_frame, origin_label_column : origin_label_column +6])
        axes_labels = np.reshape(axes_labels, (1, len(axes_labels)))
        axes_column_names = first_data.columns[origin_label_column : origin_label_column +6]
        
        for f in cam_folders:
            print(f)
            try:
                data = pd.read_hdf(os.path.join(f, 'CollectedData_Dalton.h5'))
                data[axes_column_names] = np.repeat(axes_labels, data.shape[0], axis = 0)
                    
                data.to_csv(os.path.join(f, "CollectedData_Dalton.csv"))
                data.to_hdf(os.path.join(f, "CollectedData_Dalton.h5"), "df_with_missing", format="table", mode="w")
            except:
                continue
        