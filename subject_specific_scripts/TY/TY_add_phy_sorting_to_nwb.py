#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 08:38:34 2023

@author: daltonm
"""

from pynwb import NWBHDF5IO
from neuroconv.datainterfaces import PhySortingInterface, BlackrockRecordingInterface
import ndx_pose
from importlib import sys
import os
from pathlib import Path

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import create_nwb_copy_without_acquisition, create_nwb_copy_with_external_links_to_acquisition    

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

# Append the sorting data to the NWB file by using setting the overwrite argument to False
# import phy curation   '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003'
# base_nwb_file_pattern = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003'
# phy_path = os.path.join(os.path.dirname(base_nwb_file_pattern), 'phy_IC_2023_06_12') 
make_new_processed_file = True

nwb_acquisition_file = Path('/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003_acquisition.nwb')
ns6_file = nwb_acquisition_file.parent / nwb_acquisition_file.name.replace('_acquisition.nwb', '.ns6')
nwb_storage_folder = 'demo_nwb_files' #'' 

phy_path = nwb_acquisition_file.parent / 'phy_IC_2023_06_12' 

nwb_processed_outfile = nwb_acquisition_file.parent / nwb_storage_folder / nwb_acquisition_file.name.replace('_acquisition.nwb', '_processed_correct.nwb')
if make_new_processed_file:
    nwb_processed_infile = nwb_acquisition_file 
else:
    nwb_processed_infile = nwb_processed_outfile

dataset_id = nwb_acquisition_file.parts[-2]
if dataset_id in ['TY20210211_freeAndMoths']:
    removed_chans = [29, 33, 49, 51, 66, 80]
elif dataset_id in ['TY20210211_inHammock_night']:
    removed_chans = [33, 49]
elif 'MG2023' in dataset_id:
    removed_chans = []    

write_options = dict(write_as='units', 
                     units_name='units', 
                     units_description='sorted with spikeinterface and curated with phy')

with NWBHDF5IO(nwb_acquisition_file, 'r') as io:
    nwb = io.read()
    electrodes = nwb.electrodes.to_dataframe() 
    session_start_time = nwb.session_start_time
    es_key = [key for key in nwb.acquisition.keys() if 'Electrical' in key][0]
    raw_start_time  = nwb.acquisition[es_key].starting_time
    raw_sample_rate = nwb.acquisition[es_key].rate       
    start_idx_modifier = int(raw_start_time * raw_sample_rate)

if nwb_processed_infile != nwb_processed_outfile:
    create_nwb_copy_without_acquisition(nwb_processed_infile, nwb_processed_outfile)
    # create_nwb_copy_with_external_links_to_acquisition(nwb_processed_infile, nwb_processed_outfile)

# phy_interface = PhySortingInterface(folder_path = phy_path, 
#                                     exclude_cluster_groups='noise', 
#                                     start_index_modifier=start_idx_modifier)    
phy_interface = PhySortingInterface(folder_path = phy_path, 
                                    exclude_cluster_groups='noise')
# phy_interface.adjust_spiketrain_indices_for_starting_time(start_idx_modifier)
source_data = dict(file_path=ns6_file)
recordingInterface = BlackrockRecordingInterface(**source_data)
phy_interface.register_recording(recordingInterface)
# phy_interface.set_aligned_starting_time(aligned_starting_time=raw_start_time)
# phy_interface.set_aligned_starting_time(aligned_starting_time=raw_start_time)
phy_extractor = phy_interface.sorting_extractor

# phy_interface = PhySortingInterface(folder_path = phy_path, 
#                                     exclude_cluster_groups='noise', 
#                                     start_index_modifier=start_idx_modifier)
# phy_extractor = phy_interface.sorting_extractor

unitIDs = phy_extractor.get_unit_ids()
try:
    channel_index = phy_extractor.get_property(key = 'ch', ids = unitIDs).astype('int64')
    test = channel_index[0]
except:
    channel_index = phy_extractor.get_property(key = 'group', ids = unitIDs).astype('int64')
    test = channel_index[0]
    
if removed_chans is not None:
    tmp_ch = channel_index.copy()
    for rem_ch in removed_chans:
        tmp_ch[tmp_ch >= rem_ch] = tmp_ch[tmp_ch >= rem_ch] + 1     
    channel_index = tmp_ch

electrode_labels = [electrodes.electrode_label[chIdx] for chIdx in channel_index]
x                = [electrodes.x[chIdx] for chIdx in channel_index]
y                = [electrodes.y[chIdx] for chIdx in channel_index]
z                = [electrodes.z[chIdx] for chIdx in channel_index]

properties_to_update = dict(channel_index=channel_index,
                            electrode_label=electrode_labels,
                            x=x,
                            y=y,
                            z=z)

for key, val in properties_to_update.items():
    phy_extractor.set_property(key = key,
                               values = val,
                               ids=unitIDs, 
                               missing_value=None)

for col_to_del in ['ch', 'group', 'chan_group', 'depth', 'sh']:
    try:
        del phy_extractor._properties[col_to_del]
    except:
        continue


metadata = phy_interface.get_metadata()
metadata["NWBFile"].update(session_start_time=session_start_time)

# phy_interface.run_conversion(nwbfile_path=nwb_processed_outfile,
#                              overwrite=False,
#                              metadata=metadata,
#                              **write_options)
phy_interface.run_conversion(nwbfile_path=nwb_processed_outfile,
                             overwrite=False,
                             metadata=metadata)