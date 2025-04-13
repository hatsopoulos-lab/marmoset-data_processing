#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:08:57 2023

@author: daltonm
"""

from pynwb import NWBHDF5IO
import ndx_pose
import numpy as np
import pandas as pd
import cv2
import re
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import sys, reload
from pathlib import Path
sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import read_prb_hatlab, plot_prb

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(context='notebook', style="ticks", palette='Dark2', rc=custom_params)
dark2 = sns.color_palette("Dark2")

validate_acquisition=True
validate_processed=True
marmscode = 'TYJL'
nwb_acquisition_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/EXAMPLE_TY20210211_freeAndMoths/TY20210211_freeAndMoths-003_acquisition.nwb'
nwb_processed_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/EXAMPLE_TY20210211_freeAndMoths/TY20210211_freeAndMoths-003_processed_resorted_20230612.nwb'

kinematics_video_path = Path('/project/nicho/data/marmosets/kinematics_videos/')
videos_dir = 'EXAMPLE_all_avi_videos' # FIXME should be 'avi_videos' when running this the first time to check acquisition

date_pattern = re.compile('[0-9]{8}_')

class params:
    markers = ['hand', 'shoulder'] #FIXME should ['l-wrist', 'l-d2-mcp', 'l-d5-mcp', 'l-d2-dip', 'l-d5-dip', 'l-elbow', 'l-shoulder'] for full_marmoset_model (or change l to r for right side)
    error_marker = 'hand' # FIXME should be 'l-wrist'

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
        experiment_video_path = list(kinematics_video_path.glob(f'*{tKey.split("timestamps_")[-1]}'))[0] / marmscode / date / videos_dir
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


def plot_marker_kinematics_xyz(
        reaches_df, 
        event_data, 
        markers = ['l-wrist', 'l-d2-mcp', 'l-d5-mcp', 'l-d2-dip', 'l-d5-dip', 'l-elbow', 'l-shoulder'], 
        error_marker = 'l-wrist',
        mode='event',
    ):
        
    fig, axs = plt.subplots(4, 1, figsize=(11,9), sharex=True)
    
    if mode=='event':
        time_slice = slice(0,event_data.pose_estimation_series[markers[0]].timestamps.size)
        title = f'Event {event_data.name}'
    elif mode=='reach':
        time_slice = slice(reaches_df.start_idx, reaches_df.stop_idx)
        title = f'Reach {reaches_df.name} (Event {event_data.name})'    
        
    timestamps = event_data.pose_estimation_series[markers[0]].timestamps[time_slice]  
    reproj_error    = event_data.pose_estimation_series[error_marker].confidence[time_slice]
    for dim, dimLabel in enumerate(['x', 'y', 'z']):
        if mode == 'event':
            for ridx, reach in reaches_df.iterrows():
                axs[dim].axvspan(reach.start_time, reach.stop_time, color='k', alpha=0.25)
        
        for mlabel in markers:
            marker_kinematics = event_data.pose_estimation_series[mlabel].data[time_slice]
            axs[dim].plot(timestamps, marker_kinematics[:, dim], label=mlabel)
        axs[dim].set_ylabel(f'{dimLabel} (cm)')
    
    axs[3].plot(timestamps, reproj_error, '.')
    axs[3].set_ylabel(f'{error_marker} Reprojection Error')
    axs[3].set_xlabel('Time (sec)')
    
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.suptitle(title)


def plot_kinematics_for_inspection(
        nwb_prc, 
        markers = ['l-wrist', 'l-d2-mcp', 'l-d5-mcp', 'l-d2-dip', 'l-d5-dip', 'l-elbow', 'l-shoulder'],
        error_marker = 'l-wrist',
    ):
    
    reaches_key = [key for key in nwb_prc.intervals.keys() if 'reaching_segments' in key][0]
    reaches = nwb_prc.intervals[reaches_key].to_dataframe()
    
    for video_event in reaches['video_event'].unique():
        event_reaches_df = reaches.loc[reaches["video_event"] == video_event, :]
        event_data = nwb_prc.processing[event_reaches_df.kinematics_module.values[0]].data_interfaces[video_event]
        
        plot_marker_kinematics_xyz(event_reaches_df, event_data, markers, error_marker, mode='event')
        
        for ridx, reach in event_reaches_df.iterrows():
            plot_marker_kinematics_xyz(reach, event_data, markers, error_marker, mode='reach')


def validate_processed_file(nwb_prc, nwb_acq, plot_spikes=True, plot_kinematics=True):

    if plot_spikes:    
        plot_spiketimes_to_check_timing_and_unit_to_signal_alignment(nwb_prc, nwb_acq)    
    if plot_kinematics:
        plot_kinematics_for_inspection(nwb_prc, params.markers, params.error_marker)
    
with NWBHDF5IO(nwb_acquisition_file, mode='r') as io_acq:
    nwb_acq = io_acq.read()

    if validate_acquisition:
        elec_df = validate_acquisition_nwb(nwb_acq)

    if validate_processed:
        with NWBHDF5IO(nwb_processed_file, mode='r') as io_prc:
            nwb_prc = io_prc.read()
            
            validate_processed_file(nwb_prc, nwb_acq, plot_spikes=True, plot_kinematics=True)