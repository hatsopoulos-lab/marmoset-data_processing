#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:08:57 2023

@author: daltonm
"""

from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget
import ndx_pose
import numpy as np
import pandas as pd
import cv2
import re
import matplotlib.pyplot as plt
from importlib import sys, reload
from pathlib import Path
sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import read_prb_hatlab, plot_prb

validate_acquisition=True
validate_processed=False
marmscode = 'TYJL'
nwb_acquisition_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210311_1545_freeAndCrickets_afternoon/'
nwb_processed_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_inHammock_night/TY20210211_inHammock_night-002_processed.nwb'

kinematics_video_path = Path('/project/nicho/data/marmosets/kinematics_videos/')

date_pattern = re.compile('[0-9]{8}_')

def validate_acquisition_nwb(nwb):
    print(nwb)
    
    es_key = [key for key in nwb.acquisition.keys() if 'Electrical' in key][0]

    # create timestamps for raw neural data from starting_time, rate, and data shape
    start = nwb.acquisition[es_key].starting_time
    step = 1/nwb.acquisition[es_key].rate
    stop = start + step*nwb.acquisition[es_key].data.shape[0]
    print(f'sample rate = {nwb_acq.acquisition[es_key].rate} kHz')
    print(f'start time in neural data is {start} sec')
    print(f'stop time in neural data is {stop} sec')
    
    elec_df = nwb.electrodes.to_dataframe()
    analog_idx = [idx for idx, name in elec_df['electrode_label'].items() if 'ainp' in name]
    array_idx = [idx for idx, name in elec_df['electrode_label'].items() if 'elec' in name]
    
    print('\n-------------------------------------------------------------------\n')
    
    print(nwb.acquisition['screenshots of neural data acquisition'])
    for img_name, img in nwb.acquisition['screenshots of neural data acquisition'].images.items():
        print(f'plotting {img_name}')
        plt.imshow(img)
        plt.show()
        
    print('\n-------------------------------------------------------------------\n')
    
    print(nwb.acquisition['neural signal dropout plots'])
    for img_name, img in nwb.acquisition['neural signal dropout plots'].images.items():
        print(f'plotting {img_name}')
        plt.imshow(img, interpolation='nearest')
        plt.show()

    print('\n-------------------------------------------------------------------\n')

    print(f'Channel info:\n keys = {elec_df.columns}')
    print(f'shape = {elec_df.shape}, with analog inputs in channels {analog_idx} and\n array data in channels {array_idx}')

    print('\n-------------------------------------------------------------------\n')
    
    date = re.findall(date_pattern, nwb.identifier)[0][:8]
    date = f'{date[:4]}_{date[4:6]}_{date[6:]}'
    timestamps_keys = [key for key in nwb.processing.keys() if 'timestamps' in key]
    for tKey in timestamps_keys:
        experiment_video_path = list(kinematics_video_path.glob(f'*{tKey.split("timestamps_")[-1]}'))[0] / marmscode / date / 'avi_videos'
        print(f'\nThere are {len(nwb.processing[tKey].data_interfaces)} events stored in processing-->{tKey}')
        iKey = [key for key in nwb.intervals.keys() if key.split('video_events_')[-1] == tKey.split('video_event_timestamps_')[-1]][0] 
        for idx, (key, values) in enumerate(nwb.processing[tKey].data_interfaces.items()):
            sess  = re.findall(re.compile('_s_[0-9]'), key)[0][-1]        
            event = re.findall(re.compile('_s_[0-9]_e_[0-9]{3}'), key)[0][-3:]
            try:
                video_paths = sorted(list(experiment_video_path.glob(f'*s{sess}_e{event}*.avi')))
                test = video_paths[0]
            except:
                video_paths = sorted(list(experiment_video_path.glob(f'*session{sess}_event{event}*.avi')))
                test = video_paths[0]
                
            frameCounts = []                  
            for vid in video_paths:
                cap = cv2.VideoCapture(str(vid))
                frameCounts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            videos_match_nwb_timestamps = [True if ct == values.timestamps.size else False for ct in frameCounts]
            if all(videos_match_nwb_timestamps):
                print(f'  {key}: frame count = {values.timestamps.size:>6}, startTime = {np.round(nwb.intervals[iKey].start_time[idx], 2):>7.2f}, stopTime = {np.round(nwb.intervals[iKey].stop_time[idx], 2):>7.2f} --- All {len(frameCounts)} video frame counts match timestamps.')
            else:
                print(f'  {key}: frame count = {values.timestamps.size:>6}, startTime = {np.round(nwb.intervals[iKey].start_time[idx], 2):>7.2f}, stopTime = {np.round(nwb.intervals[iKey].stop_time[idx], 2):>7.2f} --- Video frame counts = {frameCounts}')

        dropKey = [key for key in nwb.processing.keys() if key.split('frames_')[-1] == tKey.split('timestamps_')[1]][0]
        if len(nwb.processing[dropKey].data_interfaces) == 0:
            print(f'\nNo dropped frames in any of {tKey}')
        else:
            print(f'\nFound dropped frames in {tKey}')
            for drop_mask_key, drop_mask in nwb.processing[dropKey].data_interfaces.items():
                print(f'     {drop_mask_key}:  n(dropped_frames) = {drop_mask.data.size - np.sum(drop_mask.data[:])}')
            
            
        calibration_video_path = list(kinematics_video_path.glob(f'*{tKey.split("timestamps_")[-1]}'))[0] / marmscode / date / 'calibration'
        calib_video_paths = sorted(list(calibration_video_path.glob('*.avi')))
        frameCounts = []                  
        for vid in calib_video_paths:
            cap = cv2.VideoCapture(str(vid))
            frameCounts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))        
        print(f'\nExperiment = {tKey.split("timestamps_")[-1]}. Calibration video frame counts = {frameCounts}')
        print('\n-------------------------------------------------------------------\n')

    iKey = 'neural_dropout'
    print('\nNeural Dropout intervals')
    print(nwb.intervals[iKey].to_dataframe())
    
    return elec_df

def plot_spiketimes_to_check_timing_and_unit_to_signal_alignment(nwb, nwb_acq, nSpks=3):    
    # Check around these channel indices to make sure spike times align with spikes in the raw data. 

    es_key = [key for key in nwb_acq.acquisition.keys() if 'Electrical' in key][0]
    # create timestamps for raw neural data from starting_time, rate, and data shape
    start = nwb_acq.acquisition[es_key].starting_time
    step = 1/nwb_acq.acquisition[es_key].rate
    stop = start + step*nwb_acq.acquisition[es_key].data.shape[0]
    raw_timestamps = np.arange(start, stop, step)

    # get sorted units information, extract spike_times
    units = nwb_prc.units.to_dataframe()

    unit_num = 1
    units.sort_values(by='channel_index', axis=0, ascending=True, inplace=True)
    for row, unit in units.iterrows():     
        # get sorted units information, extract spike_times
        spike_times = unit.spike_times
        
        # Get electrodes table, extract the channel index matching the desired electrode_label
        raw_elec_table = nwb_acq.acquisition[es_key].electrodes.to_dataframe()
        conversion_factor = raw_elec_table['gain_to_uV'][unit.channel_index] * nwb_acq.acquisition[es_key].conversion
        
        # Get first 200000 samples raw data for that channel index
        # raw_data_single_chan = nwb_acq.acquisition[es_key].data[:300000, int(unit.channel_index)] * conversion_factor
   
        tMod = 0 #nwb_acq.acquisition['ElectricalSeriesRaw'].starting_time
        spikes_indexed_in_raw = [np.where(np.isclose(raw_timestamps, spk_time+tMod, atol=1e-6))[0][0] for spk_time in spike_times[:3]]
        
        start_idx, stop_idx = max(spikes_indexed_in_raw[0] - 100, 0), min(spikes_indexed_in_raw[-1] + 100, len(raw_timestamps))
        raw_data_single_chan = nwb_acq.acquisition[es_key].data[start_idx:stop_idx, int(unit.channel_index)] * conversion_factor
    
        try:    
            fig, axs = plt.subplots(1, nSpks)
            for spkIdx, ax in zip(spikes_indexed_in_raw, axs):
                win_idxs = [max(spkIdx - 100, 0), min(spkIdx + 100, len(raw_timestamps))]
                ax.plot(raw_timestamps[win_idxs[0] : win_idxs[1]], raw_data_single_chan[win_idxs[0] - start_idx : win_idxs[1] - start_idx])
                ax.plot(raw_timestamps[spkIdx], raw_data_single_chan[spkIdx-start_idx], 'or')
                ax.set_xticks([raw_timestamps[spkIdx]])
                ax.set_yticks([])
                ax.set_ylabel('Voltage') if ax == axs[0] else ax.set_ylabel('')
                ax.set_xlabel('Time (s)')   
                # win_times = [max(spikes_indexed_in_raw[spkNum] - 100, 0), spikes_indexed_in_raw[spkNum] + 100]
                # axs[spkNum].plot(raw_timestamps[win_times[0] : win_times[1]], raw_data_single_chan[win_times[0] : win_times[1]])
                # axs[spkNum].plot(raw_timestamps[spikes_indexed_in_raw[spkNum]], raw_data_single_chan[spikes_indexed_in_raw[spkNum]], 'or')
                # axs[spkNum].set_xticks([raw_timestamps[spikes_indexed_in_raw[spkNum]]])
            plt.title(f'Channel_idx = {int(unit.channel_index)}, Electrode label = {unit.electrode_label}, Unit Num = {unit_num}', 
                      loc='right')
            plt.show()
            unit_num+=1
        except:
            plt.show()
            print(f'Error for channel_index = {int(unit.channel_index)}, unit_num = {unit_num}')
            unit_num+=1
            continue

def validate_processed_file(nwb, nwb_acq):

    plot_spiketimes_to_check_timing_and_unit_to_signal_alignment(nwb, nwb_acq)    
    
    
with NWBHDF5IO(nwb_acquisition_file, mode='r') as io_acq:
    nwb_acq = io_acq.read()

    if validate_acquisition:
        elec_df = validate_acquisition_nwb(nwb_acq)

    if validate_processed:
        with NWBHDF5IO(nwb_processed_file, mode='r') as io_prc:
            nwb_prc = io_prc.read()
            
            validate_processed_file(nwb_prc, nwb_acq)