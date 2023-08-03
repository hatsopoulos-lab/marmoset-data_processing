#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:46:06 2023

@author: daltonm
"""

from pynwb import NWBHDF5IO
import ndx_pose


nwb_outfile = '/project/nicho/projects/dalton/data/TY20210211_freeAndMoths-003_resorted_20230612_DM.nwb'
nwb_infile = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003_processed_resorted_20230612.nwb'

with NWBHDF5IO(nwb_infile, 'r+') as io:
    nwb = io.read()
    with NWBHDF5IO(nwb_outfile, mode='w') as export_io:
        export_io.export(src_io=io, nwbfile=nwb)