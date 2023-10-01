#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:33:43 2023

@author: daltonm
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import re
import json
import yaml
from importlib import sys, reload
sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import read_prb_hatlab, plot_prb

array_code = 'JL_01'

params_dict = {'JL_01': {'array_type': 'utah',
                         'hemisphere': 'right',
                         'xy_inter_electrode_dist': 400,
                         'mapfile': '/project/nicho/data/marmosets/array_map_files/JL_SN-7623-000063/1124-23 SN 7623-000063_exilisABswap.cmp',
                         'prbfile': '/project/nicho/data/marmosets/prbfiles/JL_01.prb',
                         'impedances': '/project/nicho/data/marmosets/array_map_files/JL_SN-7623-000063/1124-23 SN 7623-000063.txt'},
               
               'MG_01': {'array_type': 'utah',
                         'hemisphere': 'left',
                         'xy_inter_electrode_dist': 400,
                         'mapfile': '/project/nicho/data/marmosets/array_map_files/MIDGE_ARRAY-01/SN 7623-000057_ABswappedForExilis.cmp',
                         'prbfile': '/project/nicho/data/marmosets/prbfiles/MG_01.prb',
                         'impedances': '/project/nicho/data/marmosets/array_map_files/MIDGE_ARRAY-01/1085-9 SN 7623-000057.txt'},
               
               'TY_02': {'array_type': 'utah',
                         'hemisphere': 'right',
                         'xy_inter_electrode_dist': 400,
                         'mapfile': '/project/nicho/data/marmosets/array_map_files/TONYARRAY02_docs/SN 7623-000030_ABswappedForExilis.cmp',
                         'prbfile': '/project/nicho/data/marmosets/prbfiles/TY_02.prb',
                         'impedances': '/project/nicho/data/marmosets/array_map_files/TONYARRAY02_docs/1039-5 SN 7623-000030.txt'},
               
               'HM_02': {'array_type': 'nform',
                         'hemisphere': 'right',
                         'xy_inter_electrode_dist': 500,
                         'z_inter_electrode_dist': 125,
                         'mapfile': 'UNKNOWN',
                         'prbfile': '/project/nicho/data/marmosets/prbfiles/HM_02.prb'}}

def define_rotation(array_info):
    rotmat = R.from_euler('z', 90, degrees=True)
    corner = np.array([0, 9])     
    return rotmat, corner

def load_and_sort_mapfile(array_info):
    map_df = pd.read_csv(array_info['mapfile'], sep='\t',skiprows=range(13),header=(0))
    map_df = map_df.iloc[:, :5]
    map_df.columns = [col.replace('//', '') for col in map_df.columns]
    map_df.dropna(axis = 0, how = 'all', inplace=True)
    map_df.sort_values(by = ['bank', 'elec'], inplace=True, ignore_index=True)    

    return map_df

def load_impedances(array_info):
    imp_df = pd.read_csv(array_info['impedances'], sep='\t', skiprows=range(7), names = ['channel_name', 'imp'])
    for idx, info in imp_df.iterrows():
        imp_df.channel_name.iloc[idx] = info['channel_name'].replace(' ', '')
        imp_df.imp.iloc[idx] = info['imp'].replace(' ', '')

    return imp_df

def create_prb_dict(array_info, map_df, imp_df):
    prb = dict()
    prb['total_nb_channels'] = map_df.shape[0]
    prb['radius'] = 100
    prb['channel_groups'] = dict()
    for elec_idx, elec_info in map_df.iterrows():
        prb['channel_groups'][elec_idx+1] = dict()
        # prb['channel_groups'][elec_idx+1]['channels'] = [elec_info['label']]
        prb['channel_groups'][elec_idx+1]['channels'] = [int(elec_info['label'].split('elec')[-1])]
        
        electrode_location = elec_info[['col', 'row']].to_numpy(dtype=float)
    
        # rotate to match schematic on brain surface
        electrode_location = np.abs(np.round(rotmat.as_matrix()[:-1, :-1] @ (electrode_location - corner))) 
    
        electrode_location = electrode_location * array_info['xy_inter_electrode_dist']
        # electrode_location = np.concatenate((electrode_location, np.array([-1000.0])))
    
        prb['channel_groups'][elec_idx+1]['geometry'] = [list(electrode_location)]
        
        prb['channel_groups'][elec_idx+1]['imp'] = imp_df.loc[imp_df.channel_name==elec_info['label'], 'imp'].to_list()

    return prb

def write_to_prbfile(array_info, prb):
    with open(array_info['prbfile'], 'w+') as f:
        f.writelines(['total_nb_channels = %d\n' % prb['total_nb_channels'], 
                      'radius = %d\n\n' % prb['radius'],
                      'channel_groups = {\n',
                      '\t# Shank index\n'])
        for chan_group, chan_info in prb['channel_groups'].items():
            f.writelines(['\t%d:\n' % chan_group,
                          '\t\t{\n',
                          "\t\t'channels': %s,\n" % chan_info['channels'],
                          "\t\t'geometry': %s,\n" % chan_info['geometry'],
                          "\t\t'impedance': %s,\n" % chan_info['imp'],
                          '\t\t},\n'])
        f.writelines('}')    

if __name__ == "__main__":
    array_info = params_dict[array_code]
    
    rotmat, corner = define_rotation(array_info)
    map_df = load_and_sort_mapfile(array_info)  
    imp_df = load_impedances(array_info)
    prb = create_prb_dict(array_info, map_df, imp_df)
    write_to_prbfile(array_info, prb)

    probegroup, imp = read_prb_hatlab(array_info['prbfile'])
    plot_prb(probegroup)

