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

class path:
    curated_spikes = r'Z:/marmosets/electrophysArchive/TY20210211_freeAndMoths/processed/003/phy_TDC_DM_20220504'
    mapfile = r'C:/Users/Dalton/Downloads/Tony_mapfile.mat'
    ns6 = r'Z:/marmosets/electrophysArchive/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003.ns6'
    
    formatted_data_storage = r'Z:/marmosets/processed_datasets/formatted_spike_data'
    session_type = 'free_and_moths'
    
    date = '20210211'
    
class params:
    nElec = 96
    expDetector = 1
    downsample = 3
    eventDetectTime = .29
    eventDetector = eventDetectTime * 30000 / downsample 
    analogChans = [129, 130, 131]
    BeTL_chan = 2

    
def load_raw_neural_signals():
    uncurated_sort_path = os.path.split(path.curated_spikes)[0]
    filename = os.path.join(uncurated_sort_path, 'phy_TDC', 'recording.dat')
    with open(filename, mode='rb') as file: 
        dat = file.read()
    return dat

def gather_waveforms(dat, curated_spikes):
        
    waveforms = []
    for clustID, probe in zip(curated_spikes['cluster_info'].cluster_id, curated_spikes['cluster_info'].ch):
        times = curated_spikes['spike_times'][curated_spikes['spike_clusters'] == clustID]
        pre  = 15
        post = 25
        probe_wvf = np.empty((len(times), pre + post + 1))
        for idx, t in enumerate(times):
            probe_wvf[idx] = dat[probe, t - pre : t + post + 1]
        waveforms.append(probe_wvf)
        
    return waveforms

def compute_cluster_snr(dat, curated_spikes):
    
    waveforms = gather_waveforms(dat, curated_spikes)
    
    snr = []
    for wvf in waveforms:
        tmp = []    
    
    return

def load_data_files(return_raw = False):
    
    uncurated_sort_path = os.path.split(path.curated_spikes)[0]
    
    cluster_info = pd.read_csv(os.path.join(path.curated_spikes, 'cluster_info.tsv'), sep = '\t')
    spike_clusters         = np.load(os.path.join(path.curated_spikes,     'spike_clusters.npy'))
    spike_times            = np.load(os.path.join(uncurated_sort_path, 'phy_TDC', 'spike_times.npy')).flatten()
        
    nsx = NsxFile(path.ns6)
    elec_labels = [int(header['ElectrodeLabel'].replace('elec', '')) for header in nsx.extended_headers if 'elec' in header['ElectrodeLabel']]
    nsx.close()
    
    elec_map_file = h5py.File(path.mapfile, 'r')
    elec_map = np.array(elec_map_file['MAP'])
    
    curated_spikes = {'spike_clusters' : spike_clusters,
                      'cluster_info'   : cluster_info,
                      'spike_times'    : spike_times}    

    dat = load_raw_neural_signals()
    
    curated_spikes = compute_cluster_snr(dat, curated_spikes)
    
    if not return_raw:
        dat = []
    
    return curated_spikes, elec_map, elec_labels, dat
    
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

if __name__ == "__main__":

    curated_spikes, elec_map, elec_labels, dat = load_data_files()
    
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
    