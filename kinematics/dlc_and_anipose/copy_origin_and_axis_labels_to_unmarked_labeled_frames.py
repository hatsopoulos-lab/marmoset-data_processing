# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:58:36 2022

@author: Dalton
"""

"""
NOTE: Need to add a check for existing label markers, and skip in that case (line 65)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import argparse
from importlib import sys
sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/utils')
from utils import convert_string_inputs_to_int_float_or_bool

class label_copier():
    def __init__(self, args: dict) -> None:
        self.labels_dir     = args['labels_dir']
        self.input_dates    = args['dates']
        self.cams_to_copy   = args['cams']  
        self.labels_to_copy = args['labels']
        self.ref_frame      = args['frame_index']
        self.ref_event      = args['ref_event']
        self.scorer         = args['dlc_scorer']
        self.date_pattern   = args['date_pattern']
        self.event_patterns = args['event_patterns']
        self.new_events     = args['new_events']

    def get_matching_dates(self):
        label_folders = sorted(list(self.labels_dir.glob('*')))
        dates = []
        for f in label_folders:
            date = re.findall(self.date_pattern, f.name)
            if len(date) > 0:
                dates.append(date[0])
        
        unique_dates = []
        for date in dates:
            if date not in unique_dates:
                unique_dates.append(date)
        
        if self.input_dates is not None:
            unique_dates = [d for d in unique_dates if d in self.input_dates]
        self.dates_to_copy = unique_dates

    def copy_labels(self):
        for date in self.dates_to_copy:
            for cam in self.cams_to_copy:
                cam_folders = sorted(list(self.labels_dir.glob(f'*{date}*cam{cam}*')))
                cam_folders = [fold for fold in cam_folders if fold.name[-8:] != '_labeled']
                
                try:
                    event_nums = [int(re.findall(self.event_patterns[0], fold.name)[0].split('_e')[-1]) for fold in cam_folders]
                    test = event_nums[0]
                except:
                    event_nums = [int(re.findall(self.event_patterns[1], fold.name)[0].split('_event')[-1]) for fold in cam_folders]
                    
                if len(event_nums) == 0: 
                    print(f'No events found for cam {cam}')
                    continue
                
                if self.ref_event is None:
                    ref_folder_idx = 0
                    ref_folder = cam_folders[ref_folder_idx]
                else:
                    ref_folder_idx, ref_folder = [(idx, fold) for idx, (fold, event_num) in enumerate(zip(cam_folders, event_nums)) if event_num == self.ref_event][0]
                
                cam_folders_after_ref_event = [fold for idx, fold in enumerate(cam_folders) if idx >= ref_folder_idx]                 
                if self.new_events is not None:
                    cam_folders_after_ref_event = [fold for fold, event_num in zip(cam_folders, event_nums) if event_num in self.new_events]
                
                ref_data = pd.read_hdf(ref_folder / f'CollectedData_{self.scorer}.h5')

                bp_level = [level for level, name in enumerate(ref_data.columns.names) if name == 'bodyparts'][0]    
                partial_df = ref_data.loc[:, ref_data.columns.get_level_values(bp_level).isin(self.labels_to_copy)]
                
                if self.ref_frame is None:
                    frame_index = np.where((~np.isnan(partial_df)).sum(axis=1) >= 0.49*partial_df.shape[1])[0][0]
                else:
                    frame_index = self.ref_frame
                    
                lab_data = partial_df.iloc[frame_index, :]

                for f in cam_folders_after_ref_event:
                    # print(f)
                    try:
                        data = pd.read_hdf(f / f'CollectedData_{self.scorer}.h5')
                    except:
                        continue    
                    
                    # TODO Check for existing labels  

                    
                    ordered_columns = partial_df.columns 
                    try: 
                        data[ordered_columns]
                        print(data.shape[1])
                    except:
                        print(f)

                    data[ordered_columns] = np.repeat(np.expand_dims(lab_data.values, axis=0), 
                                                              data.shape[0], 
                                                              axis=0)
                    
                    data.to_csv(f / f'CollectedData_{self.scorer}.csv')
                    data.to_hdf(f / f'CollectedData_{self.scorer}.h5', 'df_with_missing')

                    

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--dlc_path", required=True, type=str,
        help="path to dlc project. E.g. '/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26'")
    ap.add_argument("-d", "--dates", nargs='+', required=True, type=str,
     	help="date(s) that need to have axes labels copied (can have multiple entries separated by spaces, or no elements if you wish to apply to all dates)")
    ap.add_argument("-c", "--cams", required=True, nargs='+', type=int,
     	help="cameras (cam number, not index) for which to copy labels")
    ap.add_argument("-l", "--labels", required=True, type=int,
     	help="number of axis labels")    
    ap.add_argument("-s", "--dlc_scorer", required=True, type=str,
     	help="name of dlc scorer")
    ap.add_argument("-f", '-frame_index', required=False, type=str,
     	help="Index from which to grab origin and axes labels. \nShould be 0 to use first frame, -1 to use last frame, or None to use the first frame at which all labels don't equal NaN")
    ap.add_argument("-r", '-ref_event', required=False, type=str,
     	help="Event number from which to grab origin and axes labels. \nShould be None to use first event for each cam, or a specific event number that matches the event number label in the folder name (not index)")
    ap.add_argument("-ne", '-new_events', required=False, type=str,
     	help="Event numbers that are new and should have copied labels. \nShould be None to copy to all events after 'event_num', or a list of specific event numbers that match the event number labels in the folder name (not index)")


    allow_debugging = True
    
    try:
        args = vars(ap.parse_args())
        args['frame_index'] = convert_string_inputs_to_int_float_or_bool(args['frame_index']) if 'frame_index' in args.keys() else None
        args['event_num']   = convert_string_inputs_to_int_float_or_bool(args[  'event_num']) if   'event_num' in args.keys() else None
    except:
        if allow_debugging:
            args = {'dlc_path'          : '/project/nicho/projects/marmosets/dlc_project_files/simple_5cams_marmoset_model-Dalton-2024-06-27',
                    'dates'             : ['2023_08_04'],
                    'cams'              : [4,],
                    'labels'            : ['origin','x','y','partition_top_left', 'partition_top_right'],
                    'dlc_scorer'        : 'Dalton',
                    'frame_index'       : 0,
                    'ref_event'         : 12,
                    'new_events'        : [12,16,34,36]}
            
                    
    args['date_pattern']  = re.compile('\d{4}_\d{2}_\d{2}')
    args['event_patterns'] = [re.compile('_e[0-9]{3}'), re.compile('_event[0-9]{3}')] 
    
    args['dates'] = args['dates']
    if len(args['dates']) == 0:
        print('no dates had axis labels copied')
    else:
        args['labels_dir'] = Path(args['dlc_path']) / 'labeled-data'
        axes_copier = label_copier(args)
        axes_copier.get_matching_dates()
        axes_copier.copy_labels()
        