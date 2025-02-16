#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:46:06 2023

@author: daltonm
"""

from pynwb import NWBHDF5IO
import ndx_pose


nwb_outfile = '/project/nicho/projects/dalton/data/MG20230416_1505_mothsAndFree-002_processed_DM.nwb'
nwb_infile = '/project/nicho/data/marmosets/electrophys_data_for_processing/MG20230416_1505_mothsAndFree/MG20230416_1505_mothsAndFree-002_processed.nwb'

with NWBHDF5IO(nwb_infile, 'r+') as io:
    nwb = io.read()
    with NWBHDF5IO(nwb_outfile, mode='w') as export_io:
        export_io.export(src_io=io, nwbfile=nwb)