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
import subprocess
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

operSystem = 'linux' # can be windows or linux

electroPath = '/home/daltonm/Documents/Lab_files/spykesim_and_foraging/data/marmoset_BeTL_data'
sleepElectroPath = '/home/daltonm/Documents/Lab_files/spykesim_and_foraging/data/marmoset_BeTL_data'
processedData_dir     = '/marmosets/processed_datasets/2021_03_29/'

subprocess.run(['sudo', 'mkdir', '-p', processedData_dir])

camExposuresFiles = ['TY20210329_1423_freeAndBeTL_afternoon.ns6']
                     # 'TY20210329_1423_freeAndBeTL_afternoon.ns6'
                     # 'TY20210330_1445_freeAndBeTL_afternoon001.ns6']

spikeFiles = ['TY20210330_1445_freeAndBeTL_afternoon001-matched2night-csv.txt']
# ['TY20210329_1423_freeAndBeTL_afternoon-mached2night-csv.txt']
# ['TY20210330_1445_freeAndBeTL_afternoon001-matched2night-csv.txt']


sleepSpikeFiles = ['TY20210330_2154_inHammock_night-matched2afternoon-csv.txt']
# ['TY20210329_2145_inHammock_night002-matched2afternoon-csv.txt']
# ['TY20210330_2154_inHammock_night-matched2afternoon-csv.txt']

camSignal_and_spikes_base_filename = '2020_03_30_BeTL_and_sleep' 

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
        
        analogSignals_processedPath   = os.path.join(r'Z:', processedData_dir, camSignal_and_spikes_base_filename) + '_analogSignals'
        spikeData_processedPath   = os.path.join(r'Z:', processedData_dir, camSignal_and_spikes_base_filename) + '_spikeData'
        sleepSpikes_processedPath = os.path.join(r'Z:', processedData_dir, camSignal_and_spikes_base_filename) + '_sleepSpikeData'
        
        del camFile, spkFile
    elif operSystem == 'linux':
        data_tmp_path = '/home/daltonm/Documents/tmpData/'
        camExp = []
        spikes = []
        for camFile, spkFile in zip(camExposuresFiles, spikeFiles):
            camExp.append(os.path.join('/media/CRI', electroPath, camFile))
            spikes.append(os.path.join('/media/CRI', electroPath, spkFile))
        sleepSpikes = []
        for sleepFile in sleepSpikeFiles:
            sleepSpikes.append(os.path.join('/media/CRI', sleepElectroPath, sleepFile))
        
        analogSignals_processedPath   = os.path.join(data_tmp_path, camSignal_and_spikes_base_filename) + '_analogSignals'
        spikeData_processedPath   = os.path.join(data_tmp_path, camSignal_and_spikes_base_filename) + '_spikeData'
        sleepSpikeData_processedPath = os.path.join(data_tmp_path, camSignal_and_spikes_base_filename) + '_sleepSpikeData'
        
        del camFile, spkFile
    
class params:
    nElec = 96
    expDetector = 1
    downsample = 3
    eventDetectTime = .29
    eventDetector = eventDetectTime * 30000 / downsample 
    analogChans = [129, 130, 131]
    BeTL_chan = 2
    
#%% Load spike data during day sessions

print('Start loading spike data')

session = []
channel = []
unit = []
timestamps = []
for fNum, f in enumerate(path.spikes):
    # open nev files
    
    print(f)
    
    if f[-3:] == 'nev':
    
        nev = NevFile(f)

        for elec in range(params.nElec):
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
                    session.append(fNum)
        nev.close()
    
    elif f[-3:] == 'mat':
        print('loading mat file')
        try:
            data = h5py.File(f, 'r')
            print('extracting pieces of data')
            spikeTimes_test = np.array(data['NEV']['Data']['Spikes']['TimeStamp']).squeeze()
            channels_test = np.array(data['NEV']['Data']['Spikes']['Electrode']).squeeze()
            units_test = np.array(data['NEV']['Data']['Spikes']['Unit']).squeeze()
            spikeSampRate_test = np.array(data['NEV']['MetaTags']['TimeRes']).squeeze()
        except:
            print('exception ocurred')
            data = loadmat(f)
            print('data loaded')
            spikeTimes, channels, units = [], [], [] 
            for array in data.values():
                if type(array) == np.ndarray: 
                    spikeTimes.extend(array[:, 2])
                    channels.extend(array[:, 0])
                    units.extend(array[:, 1])
            spikeSampRate = 1
            spikeTimes, channels, units = np.array(spikeTimes), np.array(channels, dtype = np.int16), np.array(units, dtype = np.int16)
        
        print('data loaded - processing')
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
    elif f[-3:] == 'txt':
        
        # data = np.loadtxt(f, delimiter=',')
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        
        for elec in range(params.nElec):
            print((fNum, elec))
            chanData = data[data[:, 0] == elec, 1:]
            print('done loading data for this electrode')
            if len(chanData) > 0:
                spikes = chanData[:, -1]
                unitClass = chanData[:, 0] 
                units = list(set(unitClass))
                
                for u in units:
                    unit_idxs = [idx for idx, unit in enumerate(unitClass) if unit == u]
                    t = [spikes[idx] for idx in unit_idxs]
                    channel.append(elec)
                    unit.append(u)
                    timestamps.append(np.array(t))
#                    session.append(1)
                    session.append(fNum)

print('saving spikeData')

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

if operSystem == 'linux':
    with open(path.spikeData_processedPath + '.p', 'wb') as fp:
        pickle.dump(spikeData, fp, protocol = pickle.HIGHEST_PROTOCOL)
    savemat(path.spikeData_processedPath + '.mat', mdict = spikeData4mat)
    
    print('moving spike data to CRI')
    subprocess.run(['sudo', 'mv', path.spikeData_processedPath + '.p', processedData_dir])
    subprocess.run(['sudo', 'mv', path.spikeData_processedPath + '.mat', processedData_dir])
    
elif operSystem == 'windows':    
    with open(path.spikeData_processedPath  + '.p', 'wb') as fp:
        pickle.dump(spikeData, fp, protocol = pickle.HIGHEST_PROTOCOL)
    savemat(path.spikeData_processedPath + '.mat', mdict = spikeData4mat)

#%% Load sleep spike data

print('Start loading spike data')

session = []
channel = []
unit = []
timestamps = []
for fNum, f in enumerate(path.sleepSpikes):

    print(f)
    
    if f[-3:] == 'nev':
    
        nev = NevFile(f)
    
        for elec in range(params.nElec):
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
                    session.append(fNum)
        nev.close()
    
    elif f[-3:] == 'mat':
        print('loading mat file')
        try:
            data = h5py.File(f, 'r')
            print('extracting pieces of data')
            spikeTimes_test = np.array(data['NEV']['Data']['Spikes']['TimeStamp']).squeeze()
            channels_test = np.array(data['NEV']['Data']['Spikes']['Electrode']).squeeze()
            units_test = np.array(data['NEV']['Data']['Spikes']['Unit']).squeeze()
            spikeSampRate_test = np.array(data['NEV']['MetaTags']['TimeRes']).squeeze()
        except:
            print('exception ocurred')
            data = loadmat(f)
            print('data loaded')
            spikeTimes, channels, units = [], [], [] 
            for array in data.values():
                if type(array) == np.ndarray: 
                    spikeTimes.extend(array[:, 2])
                    channels.extend(array[:, 0])
                    units.extend(array[:, 1])
            spikeSampRate = 1
            spikeTimes, channels, units = np.array(spikeTimes), np.array(channels, dtype = np.int16), np.array(units, dtype = np.int16)
        
        print('data loaded - processing')
        for e in np.unique(channels): 
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
    elif f[-3:] == 'txt':
        
        data = np.loadtxt(f, delimiter=',')
        
        for elec in range(params.nElec):
            print((fNum, elec))
            chanData = data[data[:, 0] == elec, 1:]
            print('done loading data for this electrode')
            if len(chanData) > 0:
                spikes = chanData[:, -1]
                unitClass = chanData[:, 0] 
                units = list(set(unitClass))
                
                for u in units:
                    unit_idxs = [idx for idx, unit in enumerate(unitClass) if unit == u]
                    t = [spikes[idx] for idx in unit_idxs]
                    channel.append(elec)
                    unit.append(u)
                    timestamps.append(np.array(t))
#                    session.append(1)
                    session.append(fNum)

print('saving spikeData')

# set up dict variables for saving to pickle and mat files           
sleepSpikeData = {'session': session, 'channel': channel, 'unit': unit, 'spikes': timestamps}

session4mat = np.empty((len(session),), dtype=np.object)
channel4mat = np.empty_like(session4mat)
unit4mat = np.empty_like(session4mat)
timestamps4mat = np.empty_like(session4mat)
for i in range(len(session)):
    session4mat[i]    = session[i]
    channel4mat[i]    = channel[i]
    unit4mat[i]       = unit[i]
    timestamps4mat[i] = timestamps[i]
sleepSpikeData4mat = {'session': session4mat, 'channel': channel4mat, 'unit': unit4mat, 'spikes': timestamps4mat}

if operSystem == 'linux':
    with open(path.sleepSpikeData_processedPath + '.p', 'wb') as fp:
        pickle.dump(sleepSpikeData, fp, protocol = pickle.HIGHEST_PROTOCOL)
    savemat(path.sleepSpikeData_processedPath + '.mat', mdict = sleepSpikeData4mat)
    
    print('moving spike data to CRI')
    subprocess.run(['sudo', 'mv', path.sleepSpikeData_processedPath + '.p', processedData_dir])
    subprocess.run(['sudo', 'mv', path.sleepSpikeData_processedPath + '.mat', processedData_dir])
    
elif operSystem == 'windows':    
    with open(path.sleepSpikeData_processedPath  + '.p', 'wb') as fp:
        pickle.dump(sleepSpikeData, fp, protocol = pickle.HIGHEST_PROTOCOL)
    savemat(path.sleepSpikeData_processedPath + '.mat', mdict = sleepSpikeData4mat)


#%%

print('start loading analogSignals')
# open camExp data and get samplingFreq and timeVector

# with open(os.path.join(electroPath, 'trigger_data_2021-3-29.pkl'), 'rb') as f:
#     triggerData = pickle.load(f)

with open(os.path.join(electroPath, 'trigger_data_2021-3-30.pkl'), 'rb') as f:
    triggerData = pickle.load(f)
    triggerData[:76] = []

session = []
eventBoundaries = []
eventTimes = []
signalSamples = []
signalTimes = []
for fNum, f in enumerate(path.camExp): 
    nsx = NsxFile(f)
    analogChans = nsx.getdata(elec_ids = params.analogChans, downsample = params.downsample)
    sampleRate = analogChans['samp_per_s']
    times = np.linspace(0, analogChans['data_time_s'], num = int(analogChans['data_time_s'] * sampleRate))
    signals = analogChans['data']
    
    # identify beginning and end of each event
    for expChan in range(np.shape(signals)[0]):
        expOpen_samples = np.where(signals[expChan, :] > 1000)[0]

        signalSamplesTmp = expOpen_samples[np.where(np.diff(expOpen_samples) > params.expDetector)[0] + 1]
        signalSamplesTmp = np.insert(signalSamplesTmp, 0, expOpen_samples[0])
        signalSamples.append(signalSamplesTmp)
        signalTimes.append(times[signalSamples[-1]]) 
        
        
        if expChan == params.BeTL_chan:
            event_startSamples = []
            event_endSamples   = []
            for trigSamples in triggerData:
                event_startSamples.append(signalSamplesTmp[trigSamples[0]])
                event_endSamples.append(signalSamplesTmp[trigSamples[1]])
        else:
            largeDiff = np.where(np.diff(expOpen_samples) > params.eventDetector)[0]
            if len(largeDiff) > 0:  
                event_startSamples = expOpen_samples[largeDiff + 1];
                event_startSamples = np.insert(event_startSamples, 0, expOpen_samples[0])
                event_endSamples = expOpen_samples[largeDiff] 
                event_endSamples = np.append(event_endSamples, expOpen_samples[-1])    
                
            else:
                event_startSamples = expOpen_samples[0]
                event_endSamples = expOpen_samples[-1]
        
        eventBoundaries.append(np.vstack((event_startSamples, event_endSamples)));
        eventTimes.append(times[eventBoundaries[-1]])
        
        session.append(fNum)
        
    nsx.close()

# set up dict variables for saving to pickle and mat files
camSignals = {'session': session, 'eventBoundaries': eventBoundaries, 
             'eventTimes': eventTimes, 'signalSamples': signalSamples,
             'signalTimes': signalTimes}

session4mat = np.empty((len(session),), dtype=np.object)
eventBoundaries4mat = np.empty_like(session4mat)
eventTimes4mat = np.empty_like(session4mat)
signalSamples4mat = np.empty_like(session4mat)
signalTimes4mat = np.empty_like(session4mat)
for i in range(len(session)):
    session4mat[i]          = session[i]
    eventBoundaries4mat[i]  = eventBoundaries[i]
    eventTimes4mat[i]     = eventTimes[i]
    signalSamples4mat[i]  = signalSamples[i]
    signalTimes4mat[i]    = signalTimes[i]
camSignals4mat = {'session': session4mat, 'eventBoundaries': eventBoundaries4mat, 
             'eventTimes': eventTimes4mat, 'signalSamples': signalSamples4mat,
             'signalTimes': signalTimes4mat}

if operSystem == 'linux':
    with open(path.analogSignals_processedPath + '.p', 'wb') as fp:
        pickle.dump(camSignals, fp, protocol = pickle.HIGHEST_PROTOCOL)
    savemat(path.analogSignals_processedPath + '.mat', mdict = camSignals4mat)
    subprocess.run(['sudo', 'mv', path.analogSignals_processedPath + '.p', processedData_dir])
    subprocess.run(['sudo', 'mv', path.analogSignals_processedPath + '.mat', processedData_dir])
    
elif operSystem == 'windows':    
    with open(path.analogSignals_processedPath  + '.p', 'wb') as fp:
        pickle.dump(camSignals, fp, protocol = pickle.HIGHEST_PROTOCOL)
    savemat(path.analogSignals_processedPath + '.mat', mdict = camSignals4mat)
    
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
    