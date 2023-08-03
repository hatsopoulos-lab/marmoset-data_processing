# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:21:33 2021

@author: Dalton
"""

import subprocess
import os
import deeplabcut
import toml
import argparse
import numpy as np
import glob
import shutil
import time
import re
from os.path import join as pjoin

def edit_anipose_params(config_data, config_file, categories, keys, values):
    print(type(categories), categories)
    print(type(keys), keys)
    print(type(values), values)
    print('', flush=True)
    for cat, key, val in zip(categories, keys, values):
        print(cat, key, val)
        if key is not None:
            config_data[cat][key] = val
        else:
            config_data[cat]      = val
    
    with open(config_file, 'w') as f:
        toml.dump(config_data, f)
        
    return

def calibrate(anipose_args):  

    aniposepath = anipose_args['aniposepath']
    
    param_category = ['nesting']
    param_names    = [None]
    param_values   = [0]
    
    # copy config.toml file into date directory and edit parameters, then calibrate cameras
    ani_config = pjoin(anipose_args['date_dir'], 'config.toml')
    ani_template = pjoin(aniposepath, 'config.toml')
    shutil.copy(ani_template, ani_config)    
    
    os.chdir(anipose_args['date_dir'])
    
    ani_cfg_data = toml.load(ani_config) 
    edit_anipose_params(ani_cfg_data, ani_config, 
                        param_category, 
                        param_names, 
                        param_values)

    subprocess.call(['anipose', 'calibrate'])
    
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
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--anipose_path", required=True, type=str,
        help="path to anipose project. E.g. '/project/nicho/data/marmosets/test'")
    ap.add_argument("-dt", "--dates", nargs='+', required=True, type=str,
     	help="date(s) of videos to run thru anipose (can have multiple entries separated by spaces)")
    args = vars(ap.parse_args())

    print('\n\n Beginning anipose calibration at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)

    epPattern = re.compile('_e[0-9]{3}')    
    for date in args['dates']:
        ddir = pjoin(args['anipose_path'], date)

        anipose_args = {'aniposepath' : args['anipose_path'],
                        'date_dir'    : ddir}
    
        # for key, val in anipose_args.items():
        #     print(key, ' : ', val, flush=True)
    
        calibrate(anipose_args)
        
    print('\n\n Completed anipose calibration at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)

                                