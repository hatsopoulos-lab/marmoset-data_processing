# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:58:36 2022

@author: Dalton
"""

import pandas as pd
import numpy as np
import glob
import os
import re

nCams = 4
num_axes_labels = 3
dlc_scorer = 'AshvinKumar'
date_to_copy = '2022_08_18' # can be a date in from '####_##_##' or None
labels_path = '/project/nicho/projects/bci/dlc_project_files/bci-AshvinKumar-2022-01-09/labeled-data'
#labels_path = '/home/marms/Documents/dlc_local/simple_joints_model-Dalton-2021-04-08/labeled-data'



fix_bumped_apparatus_origin_points = False
if fix_bumped_apparatus_origin_points:
    fix_date = '2021_02_11'
    start_event = 61 
    end_event = None # put None if end event is the last event

label_folders = sorted(glob.glob(os.path.join(labels_path, '*')))

date_pattern = re.compile('\d{4}_\d{2}_\d{2}')
dates = []
for f in label_folders:
    dates.append(re.findall(date_pattern, f)[0])

unique_dates = []
for date in dates:
    if date not in unique_dates:
        unique_dates.append(date)

if date_to_copy is not None:
    unique_dates = [d for d in unique_dates if d == date_to_copy]

if fix_bumped_apparatus_origin_points:
    unique_dates = [fix_date]

template_data = None
        
for date in unique_dates:
    for cam in range(1, nCams+1):
        cam_folders = sorted(glob.glob(os.path.join(labels_path, '*%s*cam%d*' % (date, cam))))
        if fix_bumped_apparatus_origin_points:
            if end_event is None:
                end_event = int(cam_folders[-1].split('event')[1][:3])
            cam_folders = [cFold for cFold in cam_folders 
                           if int(cFold.split('event')[1][:3]) >= start_event
                           and int(cFold.split('event')[1][:3]) <= end_event]
        first_data = pd.read_hdf(os.path.join(cam_folders[0], 'CollectedData_%s.h5' % dlc_scorer))
        try:
            axes_label_frame = np.where(~np.isnan(first_data.loc[:, (dlc_scorer, 'origin', 'x')]))[0][0]
            origin_label_column = np.where(first_data.columns == (dlc_scorer, 'origin', 'x'))[0][0]        
            axes_labels = np.array(first_data.iloc[axes_label_frame, origin_label_column : origin_label_column + 2*num_axes_labels])
            axes_labels = np.reshape(axes_labels, (1, len(axes_labels)))
            axes_column_names = first_data.columns[origin_label_column : origin_label_column + 2*num_axes_labels]
        except:
            origin_label_column = np.where(template_data.columns == (dlc_scorer, 'origin', 'x'))[0][0]
            axes_labels = np.full((1, 2*num_axes_labels), np.nan)
            axes_column_names = template_data.columns[origin_label_column : origin_label_column + 2*num_axes_labels]
        
        for f in cam_folders:
            print(f)
            try:
                data = pd.read_hdf(os.path.join(f, 'CollectedData_%s.h5' % dlc_scorer))
                data[axes_column_names] = np.repeat(axes_labels, data.shape[0], axis = 0)
                
                data.to_csv(os.path.join(f, "CollectedData_%s.csv" % dlc_scorer))
                data.to_hdf(os.path.join(f, "CollectedData_%s.h5" % dlc_scorer), "df_with_missing")
            except:
                continue
            
            if template_data is None:
                template_data = data
        