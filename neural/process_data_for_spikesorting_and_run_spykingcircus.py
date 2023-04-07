#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:38:45 2022

@author: daltonm
"""

import spikeinterface.extractors as se 
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
# import spikeinterface.widgets as sw
from os.path import join as pjoin

# User defined information about the data

pth = '/project/nicho/data/marmosets/electrophys_data_for_processing' # path to data directory
sess = 'TY20221024_testbattery' # directory where Ns6 file lives
file = 'TY20221024_testbattery_001.ns6' # name of NS6 file
prbfile = '/project/nicho/data/marmosets/prbfiles/TY_array.prb' # name of probe (.prb) file

bad_chans = []
load_processed_data = False 

# User defined paths to sorters
sorters_dir    = '/project/nicho/environments/matlabtools'
kilosort_path  = pjoin(sorters_dir, 'Kilosort-2.5/')
ironclust_path = pjoin(sorters_dir, 'ironclust/')
waveclus_path  = pjoin(sorters_dir, 'wave_clus/')

if __name__ == '__main__':
    
    if load_processed_data:
        # load up from save: If you need to reload the data above, you can just run this cell
        recording_cache = se.load_extractor_from_pickle(pjoin(pth, sess, 'processed/recording.pkl'))
        # check the channel properties are correct
        recording_cache.get_shared_channel_property_names()
    
    else:
        # specify path to recording
        recording_folder = pjoin(pth, sess, file)
        print('recording_folder: %s' % recording_folder, flush = True)
        # load recording
        recording = se.BlackrockRecordingExtractor(recording_folder, nsx_to_load=6)
        
        # load probe information
        probefile = prbfile
        print('probefile: ', probefile)
        recording_prb = recording.load_probe_file(probefile)
        # check that info correct properties are present (should be gain, group, location, name, and offset)
        recording_prb.get_shared_channel_property_names()
        
        # visualize probe geometry: want to see that it looks correct
        # w_elec = sw.plot_electrode_geometry(recording_prb)
        
        # remove bad channels: First time you run this skip.
        if len(bad_chans) > 0: 
            recording_rmc = st.preprocessing.remove_bad_channels(recording_prb, bad_channel_ids=bad_chans)
            # verify that all of the properties were trnasfered to the new recording object
            print('properties: ', recording_rmc.get_shared_channel_property_names())
            # verify bad channels have been removed
            print('ids: ', recording_rmc.get_channel_ids())
            
        # condition the signal for the lfp
        # lowpass filter for lfp
        recording_lp = st.preprocessing.bandpass_filter(recording_prb, freq_min=1, freq_max=350, filter_type='butter')
        # downsample
        recording_lfp = st.preprocessing.resample(recording_lp, resample_rate=1000)
        
        # bandpass filter for spikes
        recording_f = st.preprocessing.bandpass_filter(recording_prb, freq_min=350, freq_max=7500, filter_type='butter')
        
        # common median reference. First time switch input to recording_prb. Make it recording_rmc if you remove channels 
        recording_cmr = st.preprocessing.common_reference(recording_f, reference='median')
        
        # view the signal on channels. channel_id is the probe and trange is the time sample to view in seconds
        # sw.plot_timeseries(recording_cmr, channel_ids=[2, 5, 7], trange=[0, 6])
        
        # view the power spectrum of the data. Check that the filtering looks reasonable. You can also look at the
        # the raw data: recording, or the lfp: recording_lfp, or the spike data: recording_f
        # w_sp = sw.plot_spectrum(recording_prb, channels=[5])
        
        # save preprocessed data for spikes and cache recording
        recording_cache = se.CacheRecordingExtractor(recording_cmr, save_path=pjoin(pth, sess, 'processed/filtered_data.dat'))
        recording_cache.dump_to_dict()
        recording_cache.dump_to_pickle(pjoin(pth, sess, 'processed/recording.pkl'))
        # save preprocessed data for lfp
        se.CacheRecordingExtractor(recording_lfp, save_path=pjoin(pth, sess, 'processed/lfp_data.dat'))
    
    # set the paths for the sorters you want to run
    ss.Kilosort2_5Sorter.set_kilosort2_5_path(kilosort_path)
    ss.IronClustSorter.set_ironclust_path(ironclust_path)
    ss.WaveClusSorter.set_waveclus_path(waveclus_path)
    
    # check which sorters are installed
    ss.installed_sorters()
    
    # Start with spyking circus. list the parameters for the sorter
    ss.get_params_description('spykingcircus')
    
    # see what the default parameters are. These are the only parameters spikeinterface will let you modify.
    ss.get_default_params('spykingcircus')
    # set your own parameter values
    params = {'detect_sign': -1,
     'adjacency_radius': 100,
     'detect_threshold': 6,
     'template_width_ms': 3,
     'filter': False,
     'merge_spikes': True,
     'auto_merge': 0.75,
     'num_workers': 15,
     'whitening_max_elts': 1000,
     'clustering_max_elts': 10000}
    
    # run the sorter
    print('\n running spykingcircus\n')
    sorting_SC = ss.run_spykingcircus(recording_cache, 
                                      output_folder=pjoin(pth, sess, 'processed/results_sc'), 
                                      grouping_property='group',
                                      n_jobs=5,
                                      verbose=True, 
                                      **params)
    print(f'SpykingCircus found {len(sorting_SC.get_unit_ids())} units')
    
    # attempt to save sorting results in case of crash
    sorting_SC.dump_to_dict()
    sorting_SC.dump_to_pickle(pjoin(pth, sess, 'processed/sorting_sc.pkl'))