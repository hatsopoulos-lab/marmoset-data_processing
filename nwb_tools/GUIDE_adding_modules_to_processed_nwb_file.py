#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:49:20 2023

@author: daltonm
"""

from pynwb import NWBHDF5IO
from neuroconv.datainterfaces import PhySortingInterface
import ndx_pose
from importlib import sys

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import create_nwb_copy_without_acquisition    


'''
    If you are creating the processed file for the first time, you will want to use:
    
        nwb_acquisition_file  = base_nwb_file_pattern + '_acquisition.nwb'
        nwb_processed_infile  = nwb_acquisition_file 
        nwb_processed_outfile = base_nwb_file_pattern + '_processed.nwb'
    
    If you are adding a module to an existing processed nwb file, use:
        
        nwb_acquisition_file  = base_nwb_file_pattern + '_acquisition.nwb'
        nwb_processed_infile  = base_nwb_file_pattern + '_processed.nwb'
        nwb_processed_outfile = nwb_processed_infile

    Set base_nwb_file_pattern to the correct file for which you are adding a module, just before the _acquisition or _processed tag. For example:
        base_nwb_file_pattern = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003'
'''    
base_nwb_file_pattern = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003'
nwb_acquisition_file  = base_nwb_file_pattern + '_acquisition.nwb'
nwb_processed_infile  = base_nwb_file_pattern + '_processed.nwb'
nwb_processed_outfile = nwb_processed_infile

def add_a_module_and_data(nwb_infile, nwb_outfile, data):
    
    '''
        IMPORTANT: Before testing out new module additions, please make a copy of the input files you are using. 
                   You can copy using "cp" in the terminal (untested) or using
                   '/project/nicho/projects/marmosets/code_database/data_processing/neural/copy_nwbfile.py'
        
        Below is the outline for adding modules to the _processed.nwb file. If you are working from the original acquisition file,
        you create a copy with the acquisition portions cleared, then proceed. If adding to an existing processed file, you skip that step.
        
        Best practice for appending modules is to temporarily open the file in append mode, as shown below, and write the 
        added module within the "with () as io" statement. 
        
        First the module has to be added. Second, data has to be added to the new module using appropriate formatting. Below are
        locations to add expected new modules:
            LFP --> TimeSeries object in processing['ecephys'] module
                An example can be found in the timestamps_to_nwb and store_drop_records functions
                in '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/hatlab_nwb_functions.py'.
                A more complicated version is in 
                '/project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/pose_and_reach_data_to_nwb.py'
            
            Touchscreen trial times --> TimeIntervals object
                A simple example can be found in
                '/project/nicho/projects/marmosets/code_database/data_processing/neural/neural_dropout_first_pass.py'
                A more complicated example can be found in                 
                '/project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/pose_and_reach_data_to_nwb.py'
    '''
    
    if nwb_infile != nwb_outfile:
        create_nwb_copy_without_acquisition(nwb_infile, nwb_outfile)
    
    with NWBHDF5IO(nwb_outfile, 'r+') as io:
        nwb = io.read()
        
        data_for_module = prep_data(data)
        
        module_handle = add_module_to_nwb(module_details, module_name)
        
        add_data_to_module(module_handle, data_for_module)
        
        io.write(nwb)
        
if __name__ == '__main__':
    add_a_module_and_data(nwb_processed_infile, nwb_processed_outfile)