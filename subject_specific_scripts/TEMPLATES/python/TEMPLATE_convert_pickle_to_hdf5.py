#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:22:46 2023

@author: daltonm
"""

import dill
from pathlib import Path
from importlib import sys

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import save_dict_to_hdf5

pklpath = Path('/project/nicho/data/marmosets/processed_datasets/reach_and_trajectory_information/20230416_reach_and_trajectory_info.pkl')
savepath = pklpath.with_suffix('.h5')

with open(str(pklpath), 'rb') as f:
    reach_data = dill.load(f)
    
# save_dict_to_hdf5(results_dict, pklpath.with_suffix('.h5'))
save_dict_to_hdf5(reach_data, savepath, first_level_key='reaching_event_idx')


# results_dict_loaded = load_dict_from_hdf5(pklpath.with_suffix('.h5'), top_level_list=False, convert_4d_array_to_list = True)