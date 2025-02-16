#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:27:47 2020

@author: daltonm
"""

##### Need to test the two versions of mat files to see if they are providing the same information

import cv2
import numpy as np
import pandas as pd
# from pandas import HDFStore
# from brpylib import NevFile, NsxFile
import dill
# import matplotlib.pyplot as plt
# import shutil
# import h5py
import subprocess
from pynwb import NWBHDF5IO
from scipy.io import savemat, loadmat
import os
import glob
import re
import time
from pathlib import Path
# from scipy.signal import savgol_filter
# from pynwb import NWBFile, NWBHDF5IO, TimeSeries, behavior
# from pynwb.epoch import TimeIntervals
# from ndx_pose import PoseEstimationSeries, PoseEstimation
# import datetime
import argparse
from itertools import product

from importlib import sys
sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import timestamps_to_nwb, store_drop_records, get_electricalseries_from_nwb

session_pattern        = re.compile('_s[0-9]{1,2}')
session_pattern_backup = re.compile('_session[0-9]{1,2}')
event_pattern          = re.compile('_e[0-9]{3,5}_')
cam_pattern            = re.compile('cam[0-9]{1}.avi')
cam_pattern_backup     = re.compile('cam[0-9]{1}_filtered.avi')
date_pattern           = re.compile('/[a-zA-Z]{2,4}\d{8}_')

class params:
    
    expDetector = 1
    camSignal_voltRange = [2900, 3000]
    break_detector = .06 * 30000 
    analogChans = [129, 130, 131]
    free_chans = [1]
    app_chans = [0]
    BeTL_chans = [2]
    num_app_cams = 5
    num_free_cams = 4
    nsx_filetype = 'ns6'

    minimum_free_session_minutes = 5

def get_filepaths(ephys_path, kin_path, marms_ephys_code, marms_kin_code, date):

    date = date.replace('_', '')    

    ephys_folders = sorted(glob.glob(os.path.join(ephys_path, marms_ephys_code + '*')))
    ephys_folders = [fold for fold in ephys_folders 
                     if re.findall(datePattern, os.path.basename(fold))[0] == date
                     and any(exp.lower() in os.path.basename(fold).lower() for exp in experiments)]    
    print(ephys_folders)
    kin_outer_folders = sorted(glob.glob(os.path.join(kin_path, '*')))
    kin_outer_folders = [fold for fold in kin_outer_folders if any(exp in os.path.basename(fold).lower() for exp in experiments)]
    kin_folders = []
    for outFold in kin_outer_folders:
        inner_folders = glob.glob(os.path.join(outFold, marms_kin_code, '*'))
        weird_folders = [fold for fold in inner_folders if '.toml' not in fold and len(os.path.basename(fold).replace('_', '')) > 8]
        if len(weird_folders) > 0:
            print('These are weird folders. They will be processed but you should take note of them in case you want to delete the processed data')
            print(weird_folders)
            
        inner_folders = [fold for fold in inner_folders if '.toml' not in fold and os.path.basename(fold).replace('_', '')[:8] == date]
        inner_folders = [fold.replace('\\', '/') for fold in inner_folders]
        kin_folders.extend(inner_folders)
        
    return ephys_folders, kin_folders    


    
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
    
    debugging = False
    
    if not debugging:
    
        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
    
        ap.add_argument("-v", "--vid_dir", required=True, type=str,
            help="path to directory holding kinematic data. E.g. /project/nicho/data/marmosets/kinematics_videos")
        ap.add_argument("-ep", "--ephys_path", required=True, type=str,
            help="path to directory holding ephys data. E.g. /project/nicho/data/marmosets/electrophys_data_for_processing")
        ap.add_argument("-m", "--marms", required=True, type=str,
         	help="marmoset 4-digit code, e.g. 'JLTY'")
        ap.add_argument("-me", "--marms_ephys", required=True, type=str,
         	help="marmoset 2-digit code for ephys data, e.g. 'TY'")
        ap.add_argument("-d", "--date", required=True, type=str,
         	help="date of recording in format YYYY_MM_DD")
        ap.add_argument("-e", "--exp_name", required=True, type=str,
         	help="experiment name, e.g. free, foraging, BeTL, crickets, moths, etc")
        ap.add_argument("-e2", "--other_exp_name", required=True, type=str,
         	help="experiment name, e.g. free, foraging, BeTL, crickets, moths, etc")    
        ap.add_argument("-np", "--neur_proc_path", required=True, type=str,
         	help="path to directory holding neural processing code")
        ap.add_argument("-meta", "--meta_path", required=True, type=str,
            help="path to metadata yml file to be added to NWB file, e.g. /project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/marms_complete_metadata.yml")
        ap.add_argument("-prb", "--prb_path" , required=True, type=str,
            help="path to .prb file that provides probe/channel info to NWB file, e.g. /project/nicho/data/marmosets/prbfiles/MG_array.prb")
        ap.add_argument("-ab", "--swap_ab" , required=True, type=str,
            help="Can be 'yes' or 'no'. Indicates whether or not channel names need to be swapped for A/B bank swapping conde by exilis. For new data, this should be taken care of in cmp file. For TY data, 'yes' should be indicated.")

        args = vars(ap.parse_args())

    else:
        args = {'vid_dir'          : '/project/nicho/data/marmosets/kinematics_videos',
                'ephys_path'       : '/project/nicho/data/marmosets/electrophys_data_for_processing',
                'marms'            : 'TYJL',
                'marms_ephys'      : 'TY',
                'date'             : '2021_02_06',
                'exp_name'         : 'moth',
                'other_exp_name'   : 'free',
                'neur_proc_path'   : '/project/nicho/projects/marmosets/code_database/data_processing/neural',
                'meta_path'        : '/project/nicho/data/marmosets/metadata_yml_files/JL_complete_metadata.yml',
                'prb_path'         : '/project/nicho/data/marmosets/prbfiles/TY_02.prb',
                'swap_ab'          : 'yes',
                'debugging'        : True}
    
    try:
        task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        last_task = int(os.getenv('SLURM_ARRAY_TASK_MAX'))
    except:
        task_id = 0
        last_task = task_id
            
    if task_id == last_task:    

        print('\n\n Beginning process_analog_signals_for_episode_times.py at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
        
        datePattern = re.compile('[0-9]{8}')         
        nsx_filetype = 'ns6'
    
        experiments = [args['exp_name'], args['other_exp_name']]
        print(args['ephys_path'])
        ephys_folders, kin_folders = get_filepaths(args['ephys_path'], args['vid_dir'], args['marms_ephys'], args['marms'], args['date'])    
        print(ephys_folders)
        for eFold in ephys_folders:
            
            analogfiles = sorted(glob.glob(os.path.join(eFold, '*.%s' % params.nsx_filetype)))
            nwbfiles = [an_path.replace('.ns6', '_acquisition.nwb') for an_path in analogfiles]
            
            for nwbfile_path, nsx_path in zip(nwbfiles, analogfiles):
                print('Creating nwb file at %s' % nwbfile_path)
                subprocess.call(['python',
                                  os.path.join(args['neur_proc_path'], 'store_neural_data_in_nwb.py'),
                                  '-f', nsx_path,
                                  '-m', args['meta_path'],
                                  '-p', args['prb_path'],
                                  '-ab', args['swap_ab']])
