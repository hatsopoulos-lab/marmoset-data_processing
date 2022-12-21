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
import argparse

def copy_labels_to_unmarked_frames(labels_path, ncams, num_axes_labels, dlc_scorer, dates_to_copy):

    label_folders = sorted(glob.glob(os.path.join(labels_path, '*')))
    
    dates = []
    for f in label_folders:
        dates.append(re.findall(date_pattern, f)[0])
    

    unique_dates = []
    for date in dates:
        if date not in unique_dates:
            unique_dates.append(date)
    
    if dates_to_copy is not None:
        unique_dates = [d for d in unique_dates if d in dates_to_copy]
    
    template_data = None
            
    for date in unique_dates:
        for cam in range(1, ncams+1):
            cam_folders = sorted(glob.glob(os.path.join(labels_path, '*%s*cam%d*' % (date, cam))))
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

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--dlc_path", required=True, type=str,
        help="path to dlc project. E.g. '/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26'")
    ap.add_argument("-d", "--dates", nargs='+', required=True, type=str,
     	help="date(s) that need to have axes labels copied (can have multiple entries separated by spaces, or no elements if you wish to apply to all dates)")
    ap.add_argument("-c", "--ncams", required=True, type=int,
     	help="number of cameras")
    ap.add_argument("-l", "--nlabels", required=True, type=int,
     	help="number of axis labels")    
    ap.add_argument("-s", "--dlc_scorer", required=True, type=str,
     	help="name of dlc scorer")
     
    args = vars(ap.parse_args())

    date_pattern = re.compile('\d{4}_\d{2}_\d{2}')
    
    dates_to_copy = args['dates']
    if len(dates_to_copy) == 0:
        print('no dates had axis labels copied')
    else:
        labels_path = os.path.join(args['dlc_path'], 'labeled-data')
        copy_labels_to_unmarked_frames(labels_path, args['ncams'], args['nlabels'], args['dlc_scorer'], dates_to_copy)
    
        