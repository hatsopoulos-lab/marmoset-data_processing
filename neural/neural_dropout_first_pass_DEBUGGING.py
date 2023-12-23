#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:05:37 2022

@author: daltonm
"""

import os
import glob
from os.path import join as pjoin
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
from scipy.signal import butter, sosfilt, find_peaks
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
# from brpylib import NevFile, NsxFile
import re
import dill
from pynwb import NWBHDF5IO
from pynwb.epoch import TimeIntervals
from pynwb.image import RGBImage
from pynwb.base import Images

def time2bin(t, binwin=0.01, buffer=0.45, lastBin=False, window=0.2):
    # the default is to get the first bin of integration, but sometimes we want the last bin! \n",
    if lastBin:
        bin_idx = int((t+buffer-window)/binwin)
    else:
        bin_idx = int(((t)+(buffer))/binwin)
    return bin_idx


def bin_spikes(raster_data, startTime, endTime, binwin, binary=True, plot=True, cmap='binary_r', gauss=False, std=0.1):
    # required packages: numpy, astropy (if convolving), matplotlib (if plotting)
    binned_spks = []

    if gauss:
        kernel = Gaussian1DKernel(std)
    for n, neuron in enumerate(raster_data['spiketimes']):

        bins = np.arange(startTime, endTime, binwin)

        binned, bin_edges = np.histogram(neuron, bins)
        binned = binned.astype(np.int16)

        if binary:
            binned[binned > 0] = 1

        if gauss:
            binned = convolve(binned, kernel)
        binned_spks.append(binned)
    if plot:
        plt.imshow(binned_spks, cmap=cmap)
        if gauss:
            plt.colorbar()

    return binned_spks, bin_edges


def bin2time(bin_idx, binwin=0.01, buffer=0.45, lastBin=False, window=0.2):
    # the default is to get the first bin of integration, but sometimes we want the last bin!
    if lastBin:
        t = (bin_idx*binwin)-buffer+window
    else:
        t = (bin_idx*binwin)-buffer

    return t

# modified from: https://stackoverflow.com/questions/61760669/numpy-1d-array-find-indices-of-boundaries-of-subsequences-of-the-same-number
def first_and_last_seq(x, n):
    a = np.r_[n-1,x,n-1]
    a = a==n
    start = np.r_[False,~a[:-1] & a[1:]]
    end = np.r_[a[:-1] & ~a[1:], False]
    return [np.where(start)[0]-1, np.where(end)[0]-1]

def get_filepaths(ephys_path, kin_path, marms_ephys_code, marms_kin_code, date):

    date = date.replace('_', '')    

    ephys_folders = sorted(glob.glob(os.path.join(ephys_path, marms_ephys_code + '*')))
    ephys_folders = [fold for fold in ephys_folders 
                     if re.findall(datePattern, os.path.basename(fold))[0] == date
                     and any(exp in os.path.basename(fold).lower() for exp in experiments)]    

    kin_outer_folders = sorted(glob.glob(os.path.join(kin_path, '*')))
    kin_outer_folders = [fold for fold in kin_outer_folders if any(exp in os.path.basename(fold).lower() for exp in experiments)]
    kin_folders = []
    for outFold in kin_outer_folders:
        inner_folders = glob.glob(os.path.join(outFold, marms_kin_code, '*'))
        weird_folders = [fold for fold in inner_folders if '.toml' not in fold and len(os.path.basename(fold).replace('_', '')) > 8]
        if len(weird_folders) > 0:
            print('These are weird folders. They will be processed but you should take note of them in case you want to delete the processed data')
            print(weird_folders)
            
        inner_folders = [fold for fold in inner_folders if '.toml' not in fold and os.path.basename(fold).replace('_', '')[:8] == date]
        inner_folders = [fold.replace('\\', '/') for fold in inner_folders]
        kin_folders.extend(inner_folders)
        
    return ephys_folders, kin_folders  

def identify_dropout(filepath, binwin, dropout_method = 'spikes', plot = False):
        
    with NWBHDF5IO(filepath, 'r+') as io:
        nwbfile = io.read()

        if dropout_method == 'spikes':
            
            units = nwbfile.processing['ecephys'].data_interfaces['units_from_nevfile'].to_dataframe()
            spike_times = units['spike_times'].to_list()
            num_neurons = len(spike_times)
        
        else: 
            #TODO this method currently grabs spikes from filtered raw data, but this won't work during sleep 
            #(downstates will look like dropout). Whoever is working with sleep data will need to figure a method out for this.
            
            raw = nwbfile.acquisition['ElectricalSeries']
            elec_df = raw.electrodes.to_dataframe()
            channel_idx = [idx for idx, name in elec_df['electrode_label'].iteritems() if 'ainp' not in name]
            signals = raw.data[:, channel_idx] * elec_df['gain_to_uV'][channel_idx] * raw.conversion
                            
            start = raw.starting_time
            step = 1/raw.rate
            stop = start + step*signals.shape[0]
            timestamps = np.arange(start, stop, step)
            
            num_neurons = 96
            
            spike_times = []
            for elec in channel_idx:
            
                chan_data = signals[:, elec]
                # first_data_idx = np.where(chan_data[0] != 0)[0][0]
                # signal = raw_data['data'][0, first_data_idx:]
            
                # bandpass = butter(4, [300/downsamp, 3000/downsamp], btype='bandpass',
                #                   fs=raw_data['samp_per_s'], output='sos')
                # filtered_signal = sosfilt(bandpass, signal)
            
                # thresh = 80
                # min_isi = 0.001  # 2ms
            
                # spikes, heights = find_peaks(-1*filtered_signal,
                #                              height=thresh, distance=min_isi*raw_data['samp_per_s'])
            
                # spike_times.append(spikes/raw_data['samp_per_s'])
            
        
        data = {}
        data['binwin'] = binwin
        data['times'] = {'start': np.min([np.min(x) for x in spike_times]),
                         'end': np.max([np.max(x) for x in spike_times])}
        
        data['spiketimes'] = [np.array(spike_times[x])[np.where(np.logical_and(spike_times[x] >= data['times']['start'], 
                                                                               spike_times[x] <= data['times']['end']))[0]] for x in range(num_neurons)]
        binned_spikes, _ = bin_spikes(data, data['times']['start'], data['times']['end'], data['binwin'], plot=False)
        data['binnedSpikes'] = binned_spikes
        
        spikeSum = np.sum([x for x in data['binnedSpikes']], axis=0)
    
        startTime = data['times']['start']  # 1000
        endTime = data['times']['end']  # 1050
        startBin = time2bin(startTime, lastBin=False, binwin=data['binwin'], buffer=0)
        endBin = time2bin(endTime, lastBin=False, binwin=data['binwin'], buffer=0)
        bin_times = np.array([np.round(bin2time(x, lastBin=False, binwin=data['binwin'], buffer=0), 4)
                              for x in np.arange(startBin, endBin)])
        
        val = np.where(spikeSum > 0, 1, 0)
        drop_bins = np.ones_like(val) - val
        
        ## %% Get signal quality
        
        if plot:
        
            row = 3
            col = 1
            fig, ax = plt.subplots(row, col, sharex=True, figsize=(10, 6))
            # raster
            ax[0].set_title('Raster')
            ax[0].imshow([x[startBin:endBin] for x in data['binnedSpikes']], aspect='auto')
            sns.despine()
            
            # sum of spikes over time
            ax[1].set_title('Sum of Spikes over Time')
            ax[1].plot(spikeSum)
            ax[1].set_xlim(0, len(bin_times))
            sns.despine()
            
            # binarize spike sum over time
            ax[2].set_title('Signal')
            ax[2].plot(val)
            ax[2].set_xlim(0, len(bin_times))
            ax[2].set_xlabel('Frames (%.3f sec bins)' % data['binwin'])
            
            sns.despine()
            plt.tight_layout()
            
            plt.savefig(filepath.split('.')[0] + 'dropout_plots.png')
        

            image_file = filepath.split('.')[0] + 'dropout_plots.png'
            screenshot_images = [RGBImage(name=os.path.basename(image_file), data=plt.imread(image_file)[..., :3])]
            screenshots = Images(name='neural signal dropout plots',
                                 images=screenshot_images,
                                 description="related to 'neural_dropout' field in intervals")
            nwbfile.add_acquisition(screenshots)    
            
            io.write(nwbfile)        
        
        print(f'Fraction of Bins dropped: {(len(val)-sum(val))/len(val)}', flush = True)
        print(
            f"Fraction of Bins dropped, in seconds: {(len(val)-sum(val))*data['binwin']}s out of {len(val)*data['binwin']}s", flush=True)
        
        dropout_idx = first_and_last_seq(val,0)
        dropout_duration = [((end-start)+1)*data['binwin'] for [start,end] in zip(dropout_idx[0],dropout_idx[1])]
        plt.figure(figsize=(7,5))
        plt.title('Distribution of Dropout Lengths')
        sns.histplot(dropout_duration,log_scale=True,kde=True)
        plt.xlabel('Length of Dropout (s), log scale')
        sns.despine()
        
        signal_idx = first_and_last_seq(val,1) # to get isi, get distance between dropouts
        signal_duration = [((end-start)+1)*data['binwin'] for [start,end] in zip(signal_idx[0],signal_idx[1])]
        plt.figure(figsize=(7,5))
        plt.title('Distribution of Time Between Dropouts')
        sns.histplot(signal_duration,log_scale=True,kde=False)
        plt.xlabel('Time between Dropouts (s), log scale')
        sns.despine()
    
        drop_starts = np.array([bin_num+1 for bin_num, (value, prev_val) in enumerate(zip(drop_bins[1:], drop_bins[:-1])) if value == 1 and prev_val == 0])
        drop_ends   = np.array([bin_num+1 for bin_num, (value, prev_val) in enumerate(zip(drop_bins[1:], drop_bins[:-1])) if value == 0 and prev_val == 1])
        
        drop_intervals = pd.DataFrame(data=zip(bin_times[drop_starts], 
                                               bin_times[drop_ends], 
                                               bin_times[drop_ends] - bin_times[drop_starts]), 
                                      columns = ['drop_start_time', 'drop_end_time', 'drop_length_sec'])
        
        nwbfile = io.read()
        dropout_intervals_name = 'neural_dropout'
        if dropout_intervals_name in nwbfile.intervals.keys():
            drop_mod_already_exists = True
            dropout = nwbfile.intervals[dropout_intervals_name]
            dropout_df = dropout.to_dataframe()                    
        else:
            drop_mod_already_exists = False
            dropout = TimeIntervals(name = dropout_intervals_name,
                                    description = 'intervals of dropout in neural signal, computed in %f sec bins' % binwin)
            dropout.add_column(name="drop_duration", description="duration of dropout in seconds")
            
        for dIdx in range(drop_intervals.shape[0]):
            if not drop_mod_already_exists or not any(dropout_df.start_time == drop_intervals.drop_start_time[dIdx]):
                dropout.add_row(start_time    = drop_intervals.drop_start_time[dIdx], 
                                stop_time     = drop_intervals.drop_end_time  [dIdx], 
                                drop_duration = drop_intervals.drop_length_sec[dIdx])
        
        if dropout_intervals_name not in nwbfile.intervals.keys():
            nwbfile.add_time_intervals(dropout)   

        io.write(nwbfile)                                                          
    
    fraction_dropped = np.round(sum(drop_bins) / len(drop_bins), 8)
    
    return drop_intervals, fraction_dropped

if __name__ == '__main__':
    
    debugging = True
    
    if not debugging:
    
        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-k", "--kin_dir", required=True, type=str,
            help="path to directory for task and marmoset pair. E.g. /project/nicho/data/marmosets/kinematics_videos/")
        ap.add_argument("-ep", "--ephys_path", required=True, type=str,
            help="path to directory holding ephys data. E.g. /project/nicho/data/marmosets/electrophys_data_for_processing")
        ap.add_argument("-m", "--marms", required=True, type=str,
         	help="marmoset 4-digit code, e.g. 'JLTY'")
        ap.add_argument("-me", "--marms_ephys", required=True, type=str,
         	help="marmoset 2-digit code for ephys data, e.g. 'TY'")
        ap.add_argument("-d", "--date", required=True, type=str,
         	help="date(s) of recording (can have multiple entries separated by spaces)")
        ap.add_argument("-e", "--exp_name", required=True, type=str,
         	help="experiment name, e.g. free, foraging, BeTL, crickets, moths, etc")
        ap.add_argument("-e2", "--other_exp_name", required=True, type=str,
         	help="experiment name, e.g. free, foraging, BeTL, crickets, moths, etc") 
        args = vars(ap.parse_args())

    else:
        args = {'kin_dir' : '/project/nicho/data/marmosets/kinematics_videos',
                'ephys_path' : '/project/nicho/data/marmosets/electrophys_data_for_processing',
                'date' : '2023_08_11',
                'marms': 'JLTY',
                'marms_ephys': 'JL',
                'exp_name':'moth',
                'other_exp_name': 'moth_free'}

    use_nev = True
    binwin = 0.1
    dropout_method = 'spikes'    
    
    try:
        task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        last_task = int(os.getenv('SLURM_ARRAY_TASK_MAX'))
    except:
        task_id = 0
        last_task = task_id
    
    if task_id == last_task:
        experiments = [args['exp_name'], args['other_exp_name']]
        
        datePattern = re.compile('[0-9]{8}')         
        ephys_folders, kin_folders = get_filepaths(args['ephys_path'], args['kin_dir'], args['marms_ephys'], args['marms'], args['date'])    
        
        for eFold in ephys_folders:
            nwb_files = sorted(glob.glob(pjoin(eFold, '*_acquisition.nwb')))
            for nfile in nwb_files:
                drop_intervals, fraction_dropped = identify_dropout(nfile, binwin, dropout_method=dropout_method, plot = True)
