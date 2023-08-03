#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:27:47 2020

@author: daltonm
"""

##### Need to test the two versions of mat files to see if they are providing the same information

import numpy as np
import pandas as pd
from brpylib import NevFile, NsxFile
import matplotlib.pyplot as plt
import pickle
import h5py
from scipy.io import savemat, loadmat
import os

#
#processedData_dir     = '/marmosets/processed_datasets/foraging_and_homeCage/2019_11_26/'
#traj_dir              = '/marmosets/deeplabcut_results/PT_foraging-Dalton-2019-12-03/all_trajectories/'
#traj_storage_filename = '2019_11_26_foraging_trajectories_session_1_2_3_shuffle1_330000'
#
#class params: 
#    if operSystem == 'windows':
#        traj_path          = os.path.join(r'Z:', traj_dir)
#        traj_processedPath = os.path.join(r'Z:', processedData_dir, traj_storage_filename)
#    elif operSystem == 'linux':
#        traj_path          = os.path.join('/media/CRI', traj_dir)   
#        traj_processedPath = os.path.join('/media/CRI', processedData_dir)
#        tmpStorage         = os.path.join('/home/marmosets/Documents/tmpProcessStorage/', traj_storage_filename) 

operSystem = 'windows' # can be windows or linux

electroPath = '/marmosets/electrophysArchive/2019_10_01_thru_11_30_PT_sleep_foraging_homeCage/Home_cage_and_foraging/2019_11_26/'
sleepElectroPath = '/marmosets/electrophysArchive/2019_10_01_thru_11_30_PT_sleep_foraging_homeCage/sleep/PT_2019_11_26/'
processedData_dir     = '/marmosets/processed_datasets/2019_11_26/'

camExposuresFiles = ['PT_homeCage_and_foraging_2019_11_26001.ns2',
                     'PT_homeCage_and_foraging_2019_11_26002.ns2',
                     'PT_homeCage_and_foraging_2019_11_26003.ns2']
spikeFiles = ['PT_homeCage_and_foraging_2019_11_26001-sorteDM_array.mat',
              'PT_homeCage_and_foraging_2019_11_26002-sortedDM.mat',
              'PT_homeCage_and_foraging_2019_11_26003-sortedDM.mat']
sleepSpikeFiles = ['PT_2019_11_26_sleep_001-finalSort_DM.mat']

camSignal_and_spikes_base_filename = '2019_11_26_foraging_and_homeCage' 
sleepSpikes_base_filename = '2019_11_26_sleep_spikeData'

class path:
    if operSystem == 'windows':
        camExp = []
        spikes = []
        for camFile, spkFile in zip(camExposuresFiles, spikeFiles):
            camExp.append(os.path.join(r'Z:', electroPath, camFile))
            spikes.append(os.path.join(r'Z:', electroPath, spkFile))
        sleepSpikes = []
        for sleepFile in sleepSpikeFiles:
            sleepSpikes.append(os.path.join(r'Z:', sleepElectroPath, sleepFile))
        
        camSignal_processedPath   = os.path.join(r'Z:', processedData_dir, camSignal_and_spikes_base_filename) + '_camSignals'
        spikeData_processedPath   = os.path.join(r'Z:', processedData_dir, camSignal_and_spikes_base_filename) + '_spikeData'
        sleepSpikes_processedPath = os.path.join(r'Z:', processedData_dir, sleepSpikes_base_filename)
        
        del camFile, spkFile
    elif operSystem == 'linux':
        tmp = []
    
class params:
    nElec = 64
    expDetector = 1
    eventDetector = 500
    camChans = [137, 138, 139, 140]
    
#%% 

session = []
channel = []
unit = []
timestamps = []
for fNum, f in enumerate(path.sleepSpikes):
    # open nev files
    
    if f[-3:] == 'nev':
    
        nev = NevFile(f)
    
        # get number of units recorded by each electrode
#        session = []
#        channel = []
#        unit = []
#        timestamps = []
        for elec in range(33, params.nElec):
            print((fNum, elec))
            chanData = nev.getdata([elec])
            print('done loading data for this electrode')
            if len(chanData) > 0:
                spikes = chanData['spike_events']
                unitClass = spikes['Classification'][0] 
                units = list(set(unitClass))[:-1]
                
                for u in units:
                    unit_idxs = [idx for idx, unit in enumerate(unitClass) if unit == u]
                    t = [spikes['TimeStamps'][0][idx] for idx in unit_idxs]
                    channel.append(elec)
                    unit.append(u)
                    timestamps.append(np.array(t))
#                    session.append(1)
                    session.append(fNum)
        nev.close()
    
    elif f[-3:] == 'mat':
        try:
            data = h5py.File(f)
            spikeTimes_test = np.array(data['NEV']['Data']['Spikes']['TimeStamp']).squeeze()
            channels_test = np.array(data['NEV']['Data']['Spikes']['Electrode']).squeeze()
            units_test = np.array(data['NEV']['Data']['Spikes']['Unit']).squeeze()
            spikeSampRate_test = np.array(data['NEV']['MetaTags']['TimeRes']).squeeze()
        except:
            data = loadmat(f)
            spikeTimes, channels, units = [], [], [] 
            for array in data.values():
                if type(array) == np.ndarray: 
                    spikeTimes.extend(array[:, 2])
                    channels.extend(array[:, 0])
                    units.extend(array[:, 1])
            spikeSampRate = 1
            spikeTimes, channels, units = np.array(spikeTimes), np.array(channels, dtype = np.int16), np.array(units, dtype = np.int16)
        
        for e in np.unique(channels): 
#            elec_idxs = [idx for idx, elec in enumerate(channels) if elec == e]
            elec_idxs = np.where(channels == e)[0]
            
            if len(elec_idxs) > 1:
                channelSpikes = spikeTimes[elec_idxs]
                channelUnits = units[elec_idxs] 
                uniqueUnits = np.unique(channelUnits)
                uniqueUnits[uniqueUnits > 10] = 0
                uniqueUnits = uniqueUnits[1:]
                
                for u in uniqueUnits:
                    print((fNum, e, u))
                    unit_idxs = np.where(channelUnits == u)[0]
                    channel.append(e)
                    unit.append(u)
                    timestamps.append(channelSpikes[unit_idxs] / spikeSampRate)
                    session.append(fNum)

# set up dict variables for saving to pickle and mat files           
spikeData = {'session': session, 'channel': channel, 'unit': unit, 'spikes': timestamps}

session4mat = np.empty((len(session),), dtype=np.object)
channel4mat = np.empty_like(session4mat)
unit4mat = np.empty_like(session4mat)
timestamps4mat = np.empty_like(session4mat)
for i in range(len(session)):
    session4mat[i]    = session[i]
    channel4mat[i]    = channel[i]
    unit4mat[i]       = unit[i]
    timestamps4mat[i] = timestamps[i]
spikeData4mat = {'session': session4mat, 'channel': channel4mat, 'unit': unit4mat, 'spikes': timestamps4mat}
    
with open(os.path.join(path.storage, params.spikeData_filename)  + '.p', 'wb') as fp:
    pickle.dump(spikeData, fp, protocol = pickle.HIGHEST_PROTOCOL)
    
savemat(os.path.join(path.storage, params.spikeData_filename) + '.mat', mdict = spikeData4mat)

#%%

# open camExp data and get samplingFreq and timeVector

session = []
eventBoundaries = []
segmentTimes = []
expStartSamples = []
expStartTimes = []
for fNum, f in enumerate(path.camExp): 
    nsx = NsxFile(f)
    analogChans = nsx.getdata(elec_ids = params.camChans)
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

# set up dict variables for saving to pickle and mat files
camSignals = {'session': session, 'eventBoundaries': eventBoundaries, 
             'eventTimes': segmentTimes, 'expStartSamp': expStartSamples,
             'expStartTimes': expStartTimes}

session4mat = np.empty((len(session),), dtype=np.object)
eventBoundaries4mat = np.empty_like(session4mat)
segmentTimes4mat = np.empty_like(session4mat)
expStartSamples4mat = np.empty_like(session4mat)
expStartTimes4mat = np.empty_like(session4mat)
for i in range(len(session)):
    session4mat[i]          = session[i]
    eventBoundaries4mat[i]  = eventBoundaries[i]
    segmentTimes4mat[i]     = segmentTimes[i]
    expStartSamples4mat[i]  = expStartSamples[i]
    expStartTimes4mat[i]    = expStartTimes[i]
camSignals4mat = {'session': session4mat, 'eventBoundaries': eventBoundaries4mat, 
             'eventTimes': segmentTimes4mat, 'expStartSamp': expStartSamples4mat,
             'expStartTimes': expStartTimes4mat}


with open(os.path.join(path.storage, params.camSignal_filename)  + '.p', 'wb') as fp:
    pickle.dump(camSignals, fp, protocol = pickle.HIGHEST_PROTOCOL)

savemat(os.path.join(path.storage, params.camSignal_filename) + '.mat', mdict = camSignals4mat)
    
#%%
    
#with open(path.storage + '2019_11_26_foraging_and_homeCage_spikeData.p', 'rb') as f:
#    spikeData = pickle.load(f)
#
#with open(path.storage + '2019_11_26_foraging_and_homeCage_camSignals.p', 'rb') as f:
#    camSignals = pickle.load(f)


#%% testing, delete after this

#chan1 = nsx.getdata(elec_ids = [129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 140], start_time_s= 10, data_time_s= 100)
##chan1 = nsx.getdata(elec_ids = 'all', start_time_s = 10, data_time_s = 100)
#signals = chan1['data']
##plt.plot(signals[0, 0:100000])
    