#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:11:11 2023

@author: daltonm
"""

import matplotlib.pyplot as plt
import spikeinterface
import spikeinterface.extractors as se 
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import numpy as np
import pandas as pd
from os.path import join as pjoin
import os
import glob
#import pynwb

import matplotlib.pyplot as plt

plt.switch_backend('nbAgg')

# User defined information about the data

pth = '/project/nicho/data/marmosets/electrophys_data_for_processing' # path to data directory
sess = 'MG20230416_1505_mothsAndFree_copy' # directory where Ns6 file lives
file = 'MG20230416_1505_mothsAndFree-002.ns6' # name of NS6 file
prbfile = '/project/nicho/data/marmosets/prbfiles/MG_01_integer.prb' # name of probe (.prb) file

# User defined paths to sorters
sorters_dir    = '/project/nicho/environments/matlabtools'
kilosort_path  = pjoin(sorters_dir, 'Kilosort-2.5/')
ironclust_path = pjoin(sorters_dir, 'ironclust/')
waveclus_path  = pjoin(sorters_dir, 'wave_clus/')

# load up from save: If you need to reload the data above, you can just run this cell
try:
    recording_cache = se.load_extractor_from_pickle(pjoin(pth, sess, 'processed/recording.pkl'))
    # check the channel properties are correct
    recording_cache.get_shared_channel_property_names()
    
    print('Skip subsequent steps until you reach the "Spike Sorting" header')
    
except:
    print('The processed recording does not exist at %s.\n\n Continue on to "Load Unprocessed Data" and "Preprocessing" sections' % pjoin(pth, sess, 'processed/recording.pkl'))

    def load_recording_and_prbfile(recording_folder, prbfile):
        # load recording
        recording = se.BlackrockRecordingExtractor(recording_folder, nsx_to_load=6)
        # load probe information
        recording_prb = recording.load_probe_file(prbfile)
    
        return recording_prb
    
    # defining next few steps within a function so intermediate recording_ variables don't hold on to RAM
    def process_raw_recording(recording_prb, channels_to_remove):
        
        # condition the signal for the lfp
        # lowpass filter for lfp
        recording_lp = st.preprocessing.bandpass_filter(recording_prb, freq_min=1, freq_max=350, filter_type='butter')
        # downsample
        recording_lfp = st.preprocessing.resample(recording_lp, resample_rate=1000)
        
        # remove bad channels or set recording_rmc to unaltered recording_prb
        if len(channels_to_remove) > 0:
            channels_to_remove = [ch+1 for ch in channels_to_remove]
            recording_rmc = st.preprocessing.remove_bad_channels(recording_prb, bad_channel_ids=channels_to_remove)
        else:
            recording_rmc = recording_prb
    
        # verify that all of the properties were trnasfered to the new recording object
        print('properties: ', recording_rmc.get_shared_channel_property_names())
        # verify bad channels have been removed
        print('ids: ', recording_rmc.get_channel_ids())    
        
        # bandpass filter for spikes, then common median reference
        recording_f = st.preprocessing.bandpass_filter(recording_rmc, freq_min=350, freq_max=7500, filter_type='butter')
        recording_cmr = st.preprocessing.common_reference(recording_f, reference='median')
        
        return recording_lfp, recording_cmr

    # Load recording and probefile
    recording_folder = pjoin(pth, sess, file)
    print('recording_folder: ', recording_folder)
    print('probefile: ', prbfile)
    
    recording_prb = load_recording_and_prbfile(recording_folder, prbfile)
    
    # check that info correct properties are present (should be gain, group, location, name, and offset)
    recording_prb.get_shared_channel_property_names()
   
    # automagically attempt top find bad channels. need to then specify them in next cell by 
    # editing channels_to_remove. Enter exactly the list that is printed here (output in base-0, enter in base-0).
    st.preprocessing.remove_bad_channels(recording_prb, verbose=True)
    print('(Channels not actually removed. Edit "channels_to_remove = []" in the next cell to match this output)')

    # Specify any channels to remove: Default list of channels_to_remove should be empty.
    channels_to_remove = [15, 66, 68]
    
    recording_lfp, recording_cmr = process_raw_recording(recording_prb, channels_to_remove)

    # save preprocessed data for spikes and cache recording
    recording_cache = se.CacheRecordingExtractor(recording_cmr, save_path=pjoin(pth, sess, 'processed/filtered_data.dat'))
    recording_cache.dump_to_dict()
    print('finished saving %s' % pjoin(pth, sess, 'processed/filtered_data.dat'))
    recording_cache.dump_to_pickle(pjoin(pth, sess, 'processed/recording.pkl'))
    print('finished saving %s' % pjoin(pth, sess, 'processed/recording.pkl'))
    # save preprocessed data for lfp
    se.CacheRecordingExtractor(recording_lfp, save_path=pjoin(pth, sess, 'processed/lfp_data.dat'))
    print('finished saving %s' % pjoin(pth, sess, 'processed/lfp_data.dat'))

     
# set the paths for the sorters you want to run
ss.Kilosort2_5Sorter.set_kilosort2_5_path(kilosort_path)
ss.IronClustSorter.set_ironclust_path(ironclust_path)
ss.WaveClusSorter.set_waveclus_path(waveclus_path)

# SpykingCircus
# set your own parameter values
params = {'detect_sign': -1,
 'adjacency_radius': 100,
 'detect_threshold': 6,
 'template_width_ms': 3,
 'filter': False,
 'merge_spikes': True,
 'auto_merge': 0.75,
 'num_workers': 48,
 'whitening_max_elts': 1000,
 'clustering_max_elts': 10000}

# run the sorter
sorting_SC = ss.run_spykingcircus(recording_cache, 
                                  output_folder=pjoin(pth, sess, 'processed/results_sc'), 
                                  grouping_property='group',
                                  n_jobs=48,
                                  verbose=True, 
                                  **params)
print(f'SpykingCircus found {len(sorting_SC.get_unit_ids())} units')

# find bad channels from spyking circus output
sc_channel_folders = glob.glob(pjoin(pth, sess, 'processed/results_sc', '*'))
bad_chans = []
for chfold in sc_channel_folders:
    good_chan = os.path.exists(pjoin(chfold, 'recording', 'recording.result.hdf5'))
    if not good_chan:
        bad_chans.append(int(os.path.basename(chfold)))
bad_chans = sorted(bad_chans)

print('bad_channels from spykingcircus are %s' % bad_chans)

# If no bad channels, save sorting results in case of crash
if len(bad_chans) == 0:
    sorting_SC.dump_to_dict()
    sorting_SC.dump_to_pickle(pjoin(pth, sess, 'processed/sorting_sc.pkl'))
    
# Ironclust
# create our own paramter dict
params = {'detect_sign': -1,
 'adjacency_radius': 50,
 'adjacency_radius_out': 100,
 'detect_threshold': 3.5,
 'prm_template_name': '',
 'freq_min': 300,
 'freq_max': 8000,
 'merge_thresh': 0.7,
 'pc_per_chan': 9,
 'whiten': False,
 'filter_type': 'none',
 'filter_detect_type': 'none',
 'common_ref_type': 'trimmean',
 'batch_sec_drift': 300,
 'step_sec_drift': 20,
 'knn': 30,
 'n_jobs_bin': 48,
 'chunk_mb': 500,
 'min_count': 30,
 'fGpu': False,
 'fft_thresh': 8,
 'fft_thresh_low': 0,
 'nSites_whiten': 16,
 'feature_type': 'gpca',
 'delta_cut': 1,
 'post_merge_mode': 1,
 'sort_mode': 1,
 'fParfor': True,
 'filter': False,
 'clip_pre': 0.25,
 'clip_post': 0.75,
 'merge_thresh_cc': 1,
 'nRepeat_merge': 3,
 'merge_overlap_thresh': 0.95}

# run sorter
sorting_IC = ss.run_ironclust(recording_cache, 
                              output_folder=pjoin(pth, sess, 'processed/results_ic'), 
                              grouping_property='group', 
                              n_jobs=48, 
                              verbose=True,
                              **params)
print(f'Ironclust found {len(sorting_IC.get_unit_ids())} units')

# attempt to save sorting results in case of crash
sorting_IC.dump_to_dict()
sorting_IC.dump_to_pickle(pjoin(pth, sess, 'processed/sorting_ic.pkl'))

# Waveclus
# modify parameters
params = {'detect_threshold': 4,
 'detect_sign': -1,
 'feature_type': 'wav',
 'scales': 4,
 'min_clus': 40,
 'maxtemp': 0.251,
 'template_sdnum': 3,
 'enable_detect_filter': True,
 'enable_sort_filter': True,
 'detect_filter_fmin': 300,
 'detect_filter_fmax': 3000,
 'detect_filter_order': 4,
 'sort_filter_fmin': 300,
 'sort_filter_fmax': 3000,
 'sort_filter_order': 2,
 'mintemp': 0,
 'w_pre': 20,
 'w_post': 44,
 'alignment_window': 10,
 'stdmax': 50,
 'max_spk': 40000,
 'ref_ms': 1.5,
 'interpolation': True}

# run sorter
sorting_WC = ss.run_waveclus(recording_cache, 
                             output_folder=pjoin(pth, sess, 'processed/results_wc'), 
                             grouping_property='group', 
                             n_jobs=48, 
                             verbose=True,
                             **params)
print(f'Waveclus found {len(sorting_WC.get_unit_ids())} units')

# attempt to save sorting results in case of crash
sorting_WC.dump_to_dict()
sorting_WC.dump_to_pickle(pjoin(pth, sess, 'processed/sorting_wc.pkl'))

# Curating the sorting
# compare sorter outputs. Its important here to have your primary sorter as the first sorter. The primary sorter 
# is the sort that you will actually process and use. I currently recommend using Ironclust with the default
# parameters as your primary sorters. The following code reflects this decision. If you want to use a different
# sorter as primary, you need to adjust the code below

mcmp = sc.compare_multiple_sorters([sorting_IC, sorting_WC, sorting_SC], ['IC', 'WC', 'SC'], 
                                   spiketrain_mode='union', n_jobs=48, 
                                   verbose=True)

# visualize comaprisons
w = sw.plot_multicomp_agreement(mcmp)
w = sw.plot_multicomp_agreement_by_sorter(mcmp)

# set agreement sorter. min agreement count is the number of sorters that had to agree to count the unit. Default
# is to use 4 (as per the paper)
agreement_sorting = mcmp.get_agreement_sorting(minimum_agreement_count=3)

# get the ids of the common units
ids = agreement_sorting.get_unit_ids()
# show user common unit ids from the primary sorter
print('Common Unit IDs: ', ids)

# crucial: cache main sorter and specify location of tmp directory. The tmp directory needs to exist in your system.
# This will eat a lot of space while its processing. I would recomment that you have 2TB+ free in the tmp directory
sorting_IC_cache = se.CacheSortingExtractor(sorting_IC, pjoin(pth, sess, 'processed/ic_sort_results_cache.dat'))
sorting_IC_cache.dump_to_pickle(pjoin(pth, sess, 'processed/ic_sorting_cache.pkl'))
os.makedirs(pjoin(pth, sess, 'processed/tmp/'), exist_ok=True)
sorting_IC_cache.set_tmp_folder(pjoin(pth, sess, 'processed/tmp/'))

sorting_IC_cache = se.load_extractor_from_pickle(pjoin(pth, sess, 'processed/ic_sorting_cache.pkl'))
sorting_IC_cache.set_tmp_folder(pjoin(pth, sess, 'processed/tmp/'))

# Export the data to phy for manual curation. This should work, but if it crashes, you will need to do the 
# processing separately. Use the code cells at the end of this docuiment to do that, than rerun this cell.
st.postprocessing.export_to_phy(recording_cache, sorting_IC_cache, 
                                output_folder=pjoin(pth, sess, 'processed/phy_IC'), 
                                ms_before=0.5, 
                                ms_after=1, 
                                compute_pc_features=True, 
                                compute_amplitudes=True, 
                                max_spikes_per_unit=None, 
                                compute_property_from_recording=False, 
                                n_jobs=48, 
                                recompute_info=False, 
                                save_property_or_features=False, 
                                verbose=True)

if 'sorting_IC' not in locals():
    sorting_IC = se.load_extractor_from_pickle(pjoin(pth, sess, 'processed/sorting_ic.pkl'))

# quality metrics
import seaborn as sns
# get quality metrics
quality_metrics = st.validation.compute_quality_metrics(sorting_IC, recording_cache, 
                                                        metric_names=['firing_rate', 'isi_violation', 'snr'], 
                                                        as_dataframe=True)
# plot the data
plt.figure()
# you can change these however you want to see the values
sns.scatterplot(data=quality_metrics, x="snr", y='isi_violation')

# Decide thresholds for quality metrics and ID sites that pass criteria
snr_thresh = 5
isi_viol_thresh = 0.5
duration = recording_cache.get_num_frames()
# first get ISI violations and see ids that pass
sorting_auto = st.curation.threshold_isi_violations(sorting_IC, isi_viol_thresh, 'greater', duration)
print('#: ', len(sorting_auto.get_unit_ids()))
print('IDs: ', sorting_auto.get_unit_ids())

# now threshold on snr, and additionally remove clusters that do not pass
sorting_auto = st.curation.threshold_snrs(sorting_auto, recording_cache, snr_thresh, 'less')
print('#: ', len(sorting_auto.get_unit_ids()))
print('IDs: ', sorting_auto.get_unit_ids())
ids = sorting_auto.get_unit_ids()

# Auto label based on criteria and comparision analysis. We do that by labelling all clusters that passed our
# criteria as MUA. Then we go back and label all clusters that were found in all sorters as 'Good'(SU).
cfile = pjoin(pth, sess, 'processed/phy_IC/cluster_group.tsv')
cg = pd.read_csv(cfile, delimiter='\t')
cg.iloc[sorting_auto.get_unit_ids(), 1] = 'mua'
cg.iloc[ids, 1] = 'good'
cg.to_csv(cfile, index=False, sep='\t') # check to see if the correct units were marked
