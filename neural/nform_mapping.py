# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:12:03 2022

@author: Dalton
"""

import numpy  as np
import pandas as pd

nChan = 96

shank_arrangement = np.array([[13, 14, 15, 16], 
                              [9 , 10, 11, 12],
                              [5 , 6 , 7 , 8 ],
                              [1 , 2 , 3 , 4 ]])

layer_1 = np.array([[96, 90, 84, 78], 
                    [72, 66, 60, 54],
                    [48, 42, 36, 30],
                    [24, 18, 12, 6]])

layer_2 = layer_1 - 1
layer_3 = layer_2 - 1
layer_4 = layer_3 - 1
layer_5 = layer_4 - 1
layer_6 = layer_5 - 1

nform = np.dstack((layer_1, layer_2, layer_3, layer_4, layer_5, layer_6))

impedances = pd.read_table('Z:/marmosets/electrophysArchive/HM20220105_arrayCheckAnesthetized/impedances', 
                           names = ['Channel', 'Impedance'], 
                           skiprows=7)
impedances = impedances.iloc[:nChan, :]
impedances['elec'] = [17,  1, 18,  2, 19,  3, 20,  4, 21,  5, 22,  6, 23,  7, 24,  8, 
                      25,  9, 26, 10, 27, 11, 28, 12, 29, 13, 30, 14, 31, 15, 32, 16, 
                      49, 33, 50, 34, 51, 35, 52, 36, 53, 37, 54, 38, 55, 39, 56, 40, 
                      57, 41, 58, 42, 59, 43, 60, 44, 61, 45, 62, 46, 63, 47, 64, 48,
                      81, 65, 82, 66, 83, 67, 84, 68, 85, 69, 86, 70, 87, 71, 88, 72,
                      89, 73, 90, 74, 91, 75, 92, 76, 93, 77, 94, 78, 95, 79, 96, 80]

impedances['Channel']   = [int(chan.split('chan')[-1]) for chan in impedances['Channel']]
impedances['Impedance'] = [int(imp.split( 'kOhm')[ 0]) for imp  in impedances['Impedance']]

nform_imp = np.empty_like(nform)
for idx, data in impedances.iterrows():
    nform_imp[nform == data.elec] = data.Impedance
    
lowImp_chan_idx = np.where(nform_imp < 1000)
good_chans = nform[lowImp_chan_idx]

good_chan_loc = np.empty((len(lowImp_chan_idx[0]), 3))
for idx, (x, y, d) in enumerate(zip(lowImp_chan_idx[0], lowImp_chan_idx[1], lowImp_chan_idx[2])):
    good_chan_loc[idx] = [x, y, d]

    
    