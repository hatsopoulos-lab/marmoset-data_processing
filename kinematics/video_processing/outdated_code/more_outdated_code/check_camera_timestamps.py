# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 11:05:38 2021

@author: daltonm
"""
import numpy as np
import pandas as pd
from brpylib import NevFile, NsxFile
import matplotlib.pyplot as plt
import pickle
import h5py
from scipy.io import savemat, loadmat
import os

# operSystem = 'windows' # can be windows or linux

# electroPath = '/marmosets/electrophysArchive/2019_10_01_thru_11_30_PT_sleep_foraging_homeCage/Home_cage_and_foraging/2019_11_26/'
# sleepElectroPath = '/marmosets/electrophysArchive/2019_10_01_thru_11_30_PT_sleep_foraging_homeCage/sleep/PT_2019_11_26/'
# processedData_dir     = '/marmosets/processed_datasets/2019_11_26/'

# camExposuresFiles = ['PT_homeCage_and_foraging_2019_11_26001.ns2',
#                      'PT_homeCage_and_foraging_2019_11_26002.ns2',
#                      'PT_homeCage_and_foraging_2019_11_26003.ns2']
# spikeFiles = ['PT_homeCage_and_foraging_2019_11_26001-sorteDM_array.mat',
#               'PT_homeCage_and_foraging_2019_11_26002-sortedDM.mat',
#               'PT_homeCage_and_foraging_2019_11_26003-sortedDM.mat']
# sleepSpikeFiles = ['PT_2019_11_26_sleep_001-finalSort_DM.mat']

# camSignal_and_spikes_base_filename = '2019_11_26_foraging_and_homeCage' 
# sleepSpikes_base_filename = '2019_11_26_sleep_spikeData'

class path:
    camExp = [r'C:/Users/daltonm/Documents/Lab_Files/electrophys_working_dir/TY20210131_freeAndForaging002.ns6']
    # if operSystem == 'windows':
    #     camExp = []
    #     spikes = []
    #     for camFile, spkFile in zip(camExposuresFiles, spikeFiles):
    #         camExp.append(os.path.join(r'Z:', electroPath, camFile))
    #         spikes.append(os.path.join(r'Z:', electroPath, spkFile))
    #     sleepSpikes = []
    #     for sleepFile in sleepSpikeFiles:
    #         sleepSpikes.append(os.path.join(r'Z:', sleepElectroPath, sleepFile))
        
    #     camSignal_processedPath   = os.path.join(r'Z:', processedData_dir, camSignal_and_spikes_base_filename) + '_camSignals'
    #     spikeData_processedPath   = os.path.join(r'Z:', processedData_dir, camSignal_and_spikes_base_filename) + '_spikeData'
    #     sleepSpikes_processedPath = os.path.join(r'Z:', processedData_dir, sleepSpikes_base_filename)
        
    #     del camFile, spkFile
    # elif operSystem == 'linux':
    #     tmp = []

       
class params:
    expDetector = 1
    downsample = 3
    eventDetectTime = .29
    eventDetector = eventDetectTime * 30000 / downsample 
    camChans = [129, 130] #[129, 130]

session = []
eventBoundaries = []
segmentTimes = []
expStartSamples = []
expStartTimes = []
for fNum, f in enumerate(path.camExp): 
    nsx = NsxFile(f)
    analogChans = nsx.getdata(elec_ids = params.camChans, downsample = params.downsample)
    sampleRate = analogChans['samp_per_s']
    times = np.linspace(0, analogChans['data_time_s'], num = int(analogChans['data_time_s'] * sampleRate))
    signals = analogChans['data']
    
    # identify beginning and end of each event
    for expChan in range(np.shape(signals)[0]):
        expOpen_samples = np.where(signals[expChan, :] > 1000)[0]
        
        largeDiff = np.where(np.diff(expOpen_samples) > params.eventDetector)[0]
        if len(largeDiff) > 0:  
            event_startTimes = expOpen_samples[largeDiff + 1];
            event_startTimes = np.insert(event_startTimes, 0, expOpen_samples[0])
            event_endTimes = expOpen_samples[largeDiff] 
            event_endTimes = np.append(event_endTimes, expOpen_samples[-1])    
            
        else:
            event_startTimes = expOpen_samples[0]
            event_endTimes = expOpen_samples[-1]
        
        eventBoundaries.append(np.vstack((event_startTimes, event_endTimes)));
        segmentTimes.append(times[eventBoundaries[-1]])
        
        expStartTmp = expOpen_samples[np.where(np.diff(expOpen_samples) > params.expDetector)[0] + 1]
        expStartTmp = np.insert(expStartTmp, 0, expOpen_samples[0])
        expStartSamples.append(expStartTmp)
        expStartTimes.append(times[expStartSamples[-1]]) 
        
        session.append(fNum)
        
    nsx.close()
    
camSignals = {'session': session, 'eventBoundaries': eventBoundaries, 
             'eventTimes': segmentTimes, 'expStartSamp': expStartSamples,
             'expStartTimes': expStartTimes}

#%% Check on signals for foraging cam, chosen event

event = 8

event_lengths = []
exp_times = camSignals['expStartTimes'][1]
for event in range(camSignals['eventTimes'][1].shape[1]):
    event_edges = camSignals['eventTimes'][1][:, event]
    evt_times = [t for t in exp_times if t >= event_edges[0] and t <= event_edges[1]]
    time_diffs = np.diff(evt_times)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time_diffs)
    # event_samp_edges = camSignals['eventBoundaries'][1][:, event]
    # sig = signals[1, event_samp_edges[0]: event_samp_edges[1]]
    # ax.plot(times[event_samp_edges[0]: event_samp_edges[1]], sig)
    # ax.set_xlabel('Time (sec)')
    # ax.set_ylabel('Exposure Signal (mV)')
    # event_lengths.append(len(evt_times))
    
