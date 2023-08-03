#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:32:48 2022

@author: daltonm
"""

# import needed toolboxes
from pprint import pprint
from pathlib import Path
from neuroconv.utils import load_dict_from_file
from neuroconv.utils.json_schema import dict_deep_update
from pynwb import NWBFile, NWBHDF5IO
from pynwb.image import RGBImage
from pynwb.base import Images
from os.path import join as pjoin
from importlib import reload, sys
from brpylib import NevFile
import glob
import yaml
import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')

from HatLabNWBConverters_with_neuroconv import MarmForageNWBConverter
from hatlab_nwb_functions import *


def add_screenshots_to_nwb(nwb_path):
    with NWBHDF5IO(nwb_path, 'r+') as io:
        nwbfile = io.read()
    
        image_files = glob.glob(os.path.join(os.path.dirname(nwb_path), '*png'))
        screenshot_images = []
        for f in image_files:
            if 'plots' not in f:
                tmp_img = RGBImage(name=os.path.basename(f),
                                   data=plt.imread(f)[..., :3])
                screenshot_images.append(tmp_img)
    
        screenshots = Images(name='screenshots of neural data acquisition',
                             images=screenshot_images,
                             description='may include spike panel, single neural channel, raster, and IP cam footage')
    
        nwbfile.add_acquisition(screenshots)    
    
        io.write(nwbfile)        

    print('%s opened, edited, and written back to file. It is now closed.' % nwb_path)
    
    return

def create_nwb_and_store_raw_neural_data(ns6_path, meta_path, prb_path, swap_ab, filter_dict):
    nwb_path = ns6_path.replace('.ns6', '_acquisition.nwb')
    nev_path = ns6_path.replace('.ns6', '.nev')   
    
    stub_test= False
    
    if Path(nwb_path).is_file():
        print('\nNWB file already exists at %s' % nwb_path)    
        return
    
    nev_exists = os.path.isfile(nev_path)
    if nev_exists:
        source_data = dict(BlackrockRecordingInterfaceRaw = dict(file_path=ns6_path),
                           BlackrockSortingInterface      = dict(file_path=nev_path))
    else:
        print('\n\n\nThere is no NEV file located at %s. If you expected to find a NEV file, cancel the job and fix this error to include nev data in the NWB file.\n\n\n' % nev_path, flush=True)
        source_data = dict(BlackrockRecordingInterfaceRaw = dict(file_path=ns6_path))    

    # Initialize converter
    converter = MarmForageNWBConverter(source_data=source_data)
    
    print('Data interfaces for this converter:')
    pprint(converter.data_interface_objects, width=120)
    
    # Get metadata from source data and from file
    metadata = converter.get_metadata()
    new_metadata = load_dict_from_file(meta_path)
    metadata = dict_deep_update(metadata, new_metadata, append_list=False, remove_repeats=True)
    
    # add session description and notes
    try:
        text_file = glob.glob(pjoin(os.path.dirname(ns6_path), '*.txt'))[0]
        with open(text_file) as f:
            text  = f.read()
        with open(text_file) as f:
            lines = f.readlines() 
        metadata['NWBFile']['session_description'] = lines[0].replace('\n', '')
        metadata['NWBFile']['notes'] = text
    
    except:
        print('could not find a notes file in %s' % os.path.dirname(ns6_path))
        
    # add identifier
    metadata['NWBFile']['identifier'] = os.path.basename(ns6_path).split('.')[0]
    
    converter.validate_metadata(metadata)
    
    # load prb file with electrode positions
    arraymap, impedances = read_prb_hatlab(prb_path)
    map_df=arraymap.to_dataframe()
    map_df['imp'] = [imp for imp in impedances]
    if 'z' not in map_df.keys():
        map_df['z'] = np.repeat(-1000.0, map_df.shape[0])
    
    
    # get array and analog group information
    array_group = [group for group in metadata['Ecephys']['ElectrodeGroup'] if 'ainp' not in group['name'].lower()][0]
    analog_group = [group for group in metadata['Ecephys']['ElectrodeGroup'] if 'ainp' in group['name'].lower()][0]
    
    # modify metadata for raw_extractor
    raw_extractor = converter.data_interface_objects['BlackrockRecordingInterfaceRaw'].recording_extractor 

    chIDs = raw_extractor.get_channel_ids()
    chNames = raw_extractor._properties['channel_name']
    if swap_ab.lower() == 'yes':
        reorder = list(range(32, 64)) + list(range(0, 32)) + list(range(64, len(chNames)))
        chNames = np.array([chNames[idx] for idx in reorder])
    raw_extractor.set_property('electrode_label', chNames)
    del raw_extractor._properties['channel_name']
    
    array_chans = [ch for ch, name in zip(chIDs, chNames) if 'ainp' not in name]
    analog_chans = [ch for ch in chIDs if ch not in array_chans]
    
    raw_extractor.set_property('x', map_df['x'], ids=array_chans, missing_value=None)
    raw_extractor.set_property('y', map_df['y'], ids=array_chans, missing_value=None)
    raw_extractor.set_property('z', map_df['z'], ids=array_chans, missing_value=None)
    
    # set properties of utah array channels
    raw_extractor.set_channel_groups([array_group['name']]*len(array_chans), array_chans)
    raw_extractor.set_channel_groups([analog_group['name']]*len(analog_chans), analog_chans)
        
    raw_extractor.set_property('filtering', [filter_dict[ns6_path[-1]]]*len(chIDs))
    
    raw_extractor.set_property('brain_region', [array_group['location']]*len(array_chans), ids= array_chans, missing_value=None)
    raw_extractor.set_property('imp', [float(str(imp).replace('<=', '')) for imp in map_df['imp'].values], ids=array_chans, missing_value=None)

    # add electrode_labels to sortingExtractor
    labels_from_raw = np.array(raw_extractor._properties['electrode_label'])
    
    # nev = NevFile(nev_path)
    # headers = nev.extended_headers
    # label_headers = [head for head in headers if 'Label' in head.keys()]
    # electrode_labels = [head['Label'] for head in label_headers]
    # electrode_labels = [label if label in labels_from_raw else None for label in electrode_labels]
    # nev_chIDs = [head['ElectrodeID'] for head in label_headers if 'ElectrodeID' in head.keys()]

    if nev_exists:
        sorting_extractor = converter.data_interface_objects['BlackrockSortingInterface'].sorting_extractor
        
        unit_name = sorting_extractor._properties['unit_name']
        channel_index = np.array([int(name.split('ch')[-1].split('#')[0]) - 1 for name in unit_name])
        sorting_electrode_labels = labels_from_raw[channel_index]
        sorting_extractor.set_property('electrode_label', sorting_electrode_labels, missing_value=None)
    # match spike_trains from extractor to the nev file
    # ordered_spikes_list = []
    # for elec_id in range(1,6):
    #     ordered_spikes_list.append(nev.getdata(elec_ids=[elec_id], wave_read='noread')['spike_events']['TimeStamps'][0][:100])
    # spks = nev.getdata(elec_ids='all', wave_read='noread') #['spike_events']['TimeStamps'][0][:100]

    # nev_id_misordered = np.array(spks['spike_events']['ChannelID'])
    
    # nev.close()
    # nev_spikes_list = []
    # for nevID in nev_chIDs:
    #     print('nevID = %d' % nevID)
    #     idx = np.where(nev_id_misordered==nevID)[0]
    #     if len(idx) > 0:
    #         nev_spikes = spks['spike_events']['TimeStamps'][int(idx)]
    #         nev_spikes = [spktime-102 for spktime in nev_spikes]
    #     else:
    #         nev_spikes = []
    #     nev_spikes_list.append(np.array(nev_spikes))
        
    # extractor_spikes = []
    # for chID in chIDs:
    #     print('chID = %d' % chID)
    #     extractor_spikes.append(sorting_extractor.get_unit_spike_train(chID))
    
    # corrected_extractor_spikes = []
    # nev_idx = 0
    # for sIdx, spikes in enumerate(extractor_spikes):
    #     if nev_spikes_list[nev_idx].shape[0] == 0:
    #         corrected_extractor_spikes.append(np.array([], dtype='int64'))
    #         nev_idx += 1
    #     else:
    #         if spikes.shape[0] == 0:
    #             continue
    #         add_count = 1
    #         while spikes.shape[0] < nev_spikes_list[nev_idx].shape[0]:
    #             print(sIdx, nev_idx, add_count)

    #             spikes = np.concatenate((spikes, extractor_spikes[sIdx+add_count]))
    #             extractor_spikes[sIdx+add_count] = np.array([])
    #             add_count += 1
    #         corrected_extractor_spikes.append(spikes)
    #         nev_idx += 1
    
    # corrected_extractor_spikes = [np.sort(sp) for sp in corrected_extractor_spikes]
            
    
    # matching_channels = []
    # for chID in chIDs:
    #     first_extractor_spikes = sorting_extractor.get_unit_spike_train(chID)[:100]
    #     nev_match = []
    #     for nev_idx, nev_spikes in enumerate(nev_spikes_list):
    #         if first_extractor_spikes.shape == nev_spikes.shape and np.all(np.abs(first_extractor_spikes - nev_spikes) <= 100):
    #             nev_match.append(nev_idx)
    #     matching_channels.append(np.array(nev_match))
            
    # channels_to_remove = [ch for ch, eLabel in zip(chIDs, electrode_labels) if eLabel is None]
    # if len(raw_extractor._properties['electrode_label']) > len(chIDs[:len(raw_extractor._properties['electrode_label'])]):
    #     sorting_extractor.set_property('electrode_label', raw_extractor._properties['electrode_label'][:len(chIDs)], chIDs, missing_value=None)
    # else:
    #     sorting_extractor.set_property('electrode_label', raw_extractor._properties['electrode_label'], chIDs[:len(raw_extractor._properties['electrode_label'])], missing_value=None)

    # # build conversion options
    if nev_exists:
        conversion_options = converter.get_conversion_options()
        conversion_options['BlackrockRecordingInterfaceRaw']=dict(stub_test=stub_test)
        conversion_options['BlackrockSortingInterface']=dict(stub_test=stub_test, write_ecephys_metadata=False)
        write_options = dict(BlackrockRecordingInterfaceRaw=dict(), BlackrockSortingInterface=dict())
        write_options['BlackrockSortingInterface']['write_as'] = 'processing'
        write_options['BlackrockSortingInterface']['units_name'] = 'units_from_nevfile'
        if np.unique(sorting_electrode_labels).shape[0] < sorting_electrode_labels.shape[0]:
            write_options['BlackrockSortingInterface']['units_description'] = 'sorted online during experimental session, autogenerated by neuroconv from .nev file'
        else:
            write_options['BlackrockSortingInterface']['units_description'] = 'unsorted, autogenerated by neuroconv from .nev file'
    else:
        conversion_options = converter.get_conversion_options()
        conversion_options['BlackrockRecordingInterfaceRaw']=dict(stub_test=stub_test)
        write_options = dict(BlackrockRecordingInterfaceRaw=dict())

    # run conversion
    converter.run_conversion(
        metadata=metadata, 
        nwbfile_path=nwb_path, 
        conversion_options=conversion_options,
        write_options=write_options
    )
    
    # If the conversion was successful this should evaluate to ``True`` as the file was created.
    if Path(nwb_path).is_file():
        print('\nNWB file successfully created with raw neural signals at %s' % nwb_path)
    else:
        print('\nAn error occurred while trying to create NWB file at %s' % nwb_path)

    add_screenshots_to_nwb(nwb_path)

    return

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--ns6_path" , required=True, type=str,
        help="path to ns6 file that will be stored in new NWB file, e.g. /project/nicho/data/marmosets/electrophys_data_for_processing/TY20221024_testbattery/TY20221024_testbattery_001.ns6")
    ap.add_argument("-m", "--meta_path", required=True, type=str,
        help="path to metadata yml file to be added to NWB file, e.g. /project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/marms_complete_metadata.yml")
    ap.add_argument("-p", "--prb_path" , required=True, type=str,
        help="path to .prb file that provides probe/channel info to NWB file, e.g. /project/nicho/data/marmosets/prbfiles/MG_array.prb")
    ap.add_argument("-ab", "--swap_ab" , required=True, type=str,
        help="Can be 'yes' or 'no'. Indicates whether or not channel names need to be swapped for A/B bank swapping conde by exilis. For new data, this should be taken care of in cmp file. For TY data, will be necessary.")
    args = vars(ap.parse_args())
    
    # args = {'ns6_path' : '/project/nicho/data/marmosets/electrophys_data_for_processing/TS20230622_1335_sleep/TS20230622_1335_sleep_001.ns6',
    #         'meta_path': '/project/nicho/data/marmosets/metadata_yml_files/MG_complete_metadata.yml',
    #         'prb_path' : '/project/nicho/data/marmosets/prbfiles/MG_01.prb',
    #         'swap_ab'  : 'no'}
    
    filter_dict = {'6': 'None'}
    
    create_nwb_and_store_raw_neural_data(args['ns6_path'], args['meta_path'], args['prb_path'], args['swap_ab'], filter_dict)
