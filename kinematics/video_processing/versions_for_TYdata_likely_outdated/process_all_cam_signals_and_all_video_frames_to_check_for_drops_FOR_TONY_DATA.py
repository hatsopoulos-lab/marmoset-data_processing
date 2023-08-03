#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:27:47 2020

@author: daltonm
"""

##### Need to test the two versions of mat files to see if they are providing the same information

import numpy as np
import pandas as pd
from pandas import HDFStore
from brpylib import NevFile, NsxFile
import pickle
import matplotlib.pyplot as plt
import shutil
import h5py
import subprocess
from scipy.io import savemat, loadmat
import os
import glob
import cv2
from scipy.signal import savgol_filter

data_tmp_path = r'C:/Users/Dalton/Documents/lab_files/tmpData'

operSystem = 'windows' # can be windows or linux

processedData_dir     = '/marmosets/processed_datasets/analog_signal_and_video_frame_information'

ephys_archive_path = '/marmosets/electrophysArchive'
kin_archive_path = '/marmosets/kinematics_videos'

marmoset_ephys_code = 'TY' # could be TY, JL, TYJL, etc - look in electroPhys_archive to find this info and to find the start date to use
marmoset_kinematics_code = 'TYJL' #check in kinematics_videos directory
date_start = '20201223' # format is YYYYMMDD
experiments = ['free', 'foraging', 'betl', 'cricket', 'moth']

# bad_dates = ['20210302', '20210309', '20210318', '20210319']
bad_dates = ['20210309', '20210319']

class path:
    if operSystem == 'windows':
        base = r'Z:'
        analog_processed_dir = os.path.join(base, processedData_dir)
        
    elif operSystem == 'linux':
        base = '/media/CRI'
        analog_processed_dir = os.path.join(base, processedData_dir)
        
    ephys_folders = sorted(glob.glob(os.path.join(base, ephys_archive_path, marmoset_ephys_code + '*')))
    ephys_folders = [fold for fold in ephys_folders 
                     if int(os.path.basename(fold)[2:10]) >= int(date_start) 
                     and any(exp in os.path.basename(fold).lower() for exp in experiments)]
    
    ephys_folders = [eFold for eFold in ephys_folders if os.path.basename(eFold)[2:10] not in bad_dates]
    
    kin_outer_folders = sorted(glob.glob(os.path.join(base, kin_archive_path, '*')))
    kin_outer_folders = [fold for fold in kin_outer_folders if any(exp in os.path.basename(fold).lower() for exp in experiments)]
    kin_folders = []
    for outFold in kin_outer_folders:
        inner_folders = glob.glob(os.path.join(outFold, marmoset_kinematics_code, '*'))
        weird_folders = [fold for fold in inner_folders if len(os.path.basename(fold).replace('_', '')) > 8]
        if len(weird_folders) > 0:
            print('These are weird folders. They will be processed but you should take not of them in case you want to delete the processed data')
            print(weird_folders)
            
        inner_folders = [fold for fold in inner_folders if int(os.path.basename(fold).replace('_', '')[:8]) >= int(date_start)]
        inner_folders = [fold.replace('\\', '/') for fold in inner_folders]
        kin_folders.extend(inner_folders)
    
    data_tmp_path = data_tmp_path
    save_fileName = '_experiment_event_and_frame_time_info'
    
    del base, kin_outer_folders, weird_folders, outFold, inner_folders
    
    
class params:
    nElec = 96
    expDetector = 1
    camSignal_voltRange = [2900, 3000]
    downsample = 10
    eventDetectTime = .29
    eventDetector = eventDetectTime * 30000 / downsample
    pre_Jan17_free_eventDetector = .4975 * 30000 / downsample
    jan13_free_eventDetector = .9 * 30000 / downsample
    break_detector = .06 * 30000 / downsample
    analogChans = [129, 130, 131]
    free_chans = [0]
    app_chans = [1]
    BeTL_chans = [2]
    num_app_cams = 2
    num_free_cams = 2

#%% Load and process analog signals

print('start loading analogSignals')
# open analogData data and get samplingFreq and timeVector

# with open(os.path.join(electroPath, 'trigger_data_2021-3-29.pkl'), 'rb') as f:
#     triggerData = pickle.load(f)

# with open(os.path.join(electroPath, 'trigger_data_2021-3-30.pkl'), 'rb') as f:
#     triggerData = pickle.load(f)
#     triggerData[:76] = []


for eFold in [path.ephys_folders[44]]: #path.ephys_folders:
    date = os.path.basename(eFold)[2:10]
    print(f'working on {date}')
    matched_kinFolders = [kFold for kFold in path.kin_folders if os.path.basename(kFold)[:10].replace('_', '') == date]
    expNames = [kinFold.split('/')[-3] for kinFold in matched_kinFolders]
    
    # get frame counts for videos
    allExp_frameCounts = []
    allExp_vidPaths = []
    maxEvent_camNum = []
    for kFold, exp in zip(matched_kinFolders, expNames):
        if exp.lower() == 'free':
            num_cams = params.num_free_cams
        else:
            num_cams = params.num_app_cams
        
        vidPaths = []
        colNames = []
        for cNum in range(num_cams):
            tmp_vidPaths = glob.glob(os.path.join(kFold, f'*cam{cNum+1}.avi'))
            if int(date) == 20210223 and exp.lower() == 'betl':
                tmp_vidPaths = glob.glob(os.path.join(kFold, 'mp4_videos', f'*cam{cNum+1}.mp4'))    
            elif (int(date) == 20210223 or int(date) == 20210130) and exp.lower() == 'free':
                tmp_vidPaths = glob.glob(os.path.join(kFold, f'*cam{cNum+1}.mp4'))            
            sortStr = [] 
            for vPath in tmp_vidPaths:
                try:
                    vSess = str(int(os.path.basename(vPath).split('_s')[1][0]))
                    vEvent = str(int(os.path.basename(vPath).split('_e')[1][:3])).zfill(3)
                except:
                    vSess = str(int(os.path.basename(vPath).split('_session')[1][0]))
                    vEvent = str(int(os.path.basename(vPath).split('_event')[1][:3])).zfill(3)
                sortStr.append(f'{vSess}_{vEvent}')
            tmp_vidPaths = [vPath for (s_ev, vPath) in sorted(zip(sortStr, tmp_vidPaths), key=lambda pair: pair[0])] 
            
            vidPaths.append(tmp_vidPaths)
            colNames.append(f'cam{cNum+1}')
        
        colNames.append('nsx_count')
        vidCounts = np.array([len(v) for v in vidPaths])
        nVids = max(vidCounts)
        vidCountMatch = np.where(vidCounts == nVids)[0]
        missingEventIdxs = []
        if len(vidCountMatch) != len(vidPaths):
            try:
                events = [int(vp.split('e')[1][:3]) for vp in vidPaths[vidCountMatch[0]]]
                for v in vidPaths:
                    currentEvents = [int(vp.split('e')[1][:3]) for vp in v]
                    missingEventIdxs.append([ev-1 for ev in events if ev not in currentEvents])
            except:
                events = [int(vp.split('event')[1][:3]) for vp in vidPaths[vidCountMatch[0]]]
                for v in vidPaths:
                    currentEvents = [int(vp.split('event')[1][:3]) for vp in v]
                    missingEventIdxs.append([ev-1 for ev in events if ev not in currentEvents])      
        
        frameCounts = pd.DataFrame(np.empty((nVids, num_cams+1)), columns=colNames)                    
        for cNum, vPaths in enumerate(vidPaths):
            for vNum, vid in enumerate(vPaths):
                print((cNum, vNum))
                cap = cv2.VideoCapture(vid)
                frameCounts.iloc[vNum, cNum] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if len(missingEventIdxs) > 0:
                missEvs = missingEventIdxs[cNum]    
                for mEv in missEvs:
                    frameCounts.iloc[mEv+1:, cNum] = frameCounts.iloc[mEv:-1, cNum]
                    frameCounts.iloc[mEv, cNum] = np.nan
        
        maxEvent_camNum.append(vidCountMatch[0])
        allExp_vidPaths.append(vidPaths)
        allExp_frameCounts.append(frameCounts)    

    analogFiles = sorted(glob.glob(os.path.join(eFold, '*.ns6')))
    allExp_signalTimes = []
    eventTimes = []
    breakTimes = []
    session = []
    for fNum, f in enumerate(analogFiles):
        nsx = NsxFile(f)
        analogChans = nsx.getdata(elec_ids = params.analogChans, downsample = params.downsample)
        try:
            sampleRate = analogChans['samp_per_s']
            times = np.linspace(0, analogChans['data_time_s'], num = int(analogChans['data_time_s'] * sampleRate))
            signals = analogChans['data']
            
            # identify beginning and end of each event
            for expChan in range(signals.shape[0]):
                expOpen_samples = np.where(signals[expChan, :] > 1000)[0]
    
                if expOpen_samples.shape[0] == 0:
                    allExp_signalTimes.append(np.array([]))
                    eventTimes.append(np.array([]))
                    breakTimes.append(np.array([]))
                else:
                    signalSamplesTmp = expOpen_samples[np.where(np.diff(expOpen_samples) > params.expDetector)[0] + 1]
                    signalSamplesTmp = np.insert(signalSamplesTmp, 0, expOpen_samples[0])
                    # signalSamples.append(signalSamplesTmp)
                    allExp_signalTimes.append(times[signalSamplesTmp])        
                    
            # Un-comment once trigger data from touchscreen is provided        
                    # if expChan in params.BeTL_chans:
                    #     event_startSamples = []
                    #     event_endSamples   = []
                    #     for trigSamples in triggerData:
                    #         event_startSamples.append(signalSamplesTmp[trigSamples[0]])
                    #         event_endSamples.append(signalSamplesTmp[trigSamples[1]])
                    # else:
                    #     largeDiff = np.where(np.diff(expOpen_samples) > params.eventDetector)[0]
                    #     if len(largeDiff) > 0:  
                    #         event_startSamples = expOpen_samples[largeDiff + 1];
                    #         event_startSamples = np.insert(event_startSamples, 0, expOpen_samples[0])
                    #         event_endSamples = expOpen_samples[largeDiff] 
                    #         event_endSamples = np.append(event_endSamples, expOpen_samples[-1])    
                            
                    #     else:
                    #         event_startSamples = expOpen_samples[0]
                    #         event_endSamples = expOpen_samples[-1]
            ##############
            
            # Comment if touchscreen data is provided
                    if int(date) < 20210117 and int(date) != 20210113:
                        largeDiff = np.where(np.diff(expOpen_samples) >= params.pre_Jan17_free_eventDetector)[0]
                        smallDiff = np.where(np.logical_and(np.diff(expOpen_samples) > params.break_detector, 
                                                            np.diff(expOpen_samples) < params.pre_Jan17_free_eventDetector))[0]
                        if len(smallDiff) > 0:  
                            break_endSamples = expOpen_samples[smallDiff + 1];
                            break_startSamples = expOpen_samples[smallDiff]  
                            breakBoundaries = np.vstack((break_startSamples, break_endSamples))
                        else:
                            breakBoundaries = []
                            
                        breakTimes.append(times[breakBoundaries])
                    elif int(date) == 20210113:
                        largeDiff = np.where(np.diff(expOpen_samples) >= params.jan13_free_eventDetector)[0]
                        smallDiff = np.where(np.logical_and(np.diff(expOpen_samples) > params.break_detector, 
                                                            np.diff(expOpen_samples) < params.jan13_free_eventDetector))[0]
                        if len(smallDiff) > 0:  
                            break_endSamples = expOpen_samples[smallDiff + 1];
                            break_startSamples = expOpen_samples[smallDiff]  
                            breakBoundaries = np.vstack((break_startSamples, break_endSamples))
                        else:
                            breakBoundaries = []
                            
                        breakTimes.append(times[breakBoundaries])
                    else:
                        largeDiff = np.where(np.diff(expOpen_samples) > params.eventDetector)[0]
                    
                    if len(largeDiff) > 0:  
                        event_startSamples = expOpen_samples[largeDiff + 1];
                        event_startSamples = np.insert(event_startSamples, 0, expOpen_samples[0])
                        event_endSamples = expOpen_samples[largeDiff] 
                        event_endSamples = np.append(event_endSamples, expOpen_samples[-1])    
                        if event_endSamples[-1] == len(times):
                            event_endSamples[-1] -= 1
                        
                    else:
                        event_startSamples = expOpen_samples[0]
                        event_endSamples = expOpen_samples[-1]
                        if event_endSamples == len(times):
                            event_endSamples -= 1
                        
            ##############
                    
                    eventBoundariesTmp = np.vstack((event_startSamples, event_endSamples))
                    # eventBoundaries.append(np.vstack((event_startSamples, event_endSamples)));
                    eventTimes.append(times[eventBoundariesTmp])
                    
                session.append(fNum)
        except:
            session.append(fNum)
            eventTimes.append([])
            breakTimes.append([])
            allExp_signalTimes.append([])        
            
        nsx.close()
        
    
    numSessions = len(np.unique(session))
    chans_per_sess = int(len(allExp_signalTimes) / numSessions)
    
    numEvents = np.array([fCounts.shape[0] for fCounts in allExp_frameCounts])
    free_idx = int(np.where(np.logical_or(numEvents == 1, numEvents == np.min(numEvents)))[0])
    allExp_frameCounts[0], allExp_frameCounts[free_idx] = allExp_frameCounts[free_idx], allExp_frameCounts[0] 
    allExp_vidPaths[0], allExp_vidPaths[free_idx] = allExp_vidPaths[free_idx], allExp_vidPaths[0] 
    expNames[0], expNames[free_idx] = expNames[free_idx], expNames[0]
    maxEvent_camNum[0], maxEvent_camNum[free_idx] = maxEvent_camNum[free_idx], maxEvent_camNum[0]
    
    if int(date) == 20210301:
        tmp_camTimes = allExp_signalTimes[1]
        tmp_start_time, tmp_end_time = eventTimes[1][:, 2]
        tmp_sigTimes = tmp_camTimes[np.logical_and(tmp_camTimes >= tmp_start_time, tmp_camTimes <= tmp_end_time)]        
    
    removed_eventTimes = []
    for tmpIdx in range(2):
        signals_to_drop = [idx for idx, times in enumerate(eventTimes) 
                           if np.shape(times)[-1] == 0 
                           or (times.shape[-1] <= 2 and np.all(np.diff(times, axis=0) < 15*60))]
        
        sess_to_drop = [sess for idx, sess in enumerate(session) 
                        if idx in signals_to_drop 
                        and idx % chans_per_sess in params.free_chans]
        
        if len(np.unique(session)) - len(sess_to_drop) > 1:
            sess_to_drop_2 = [sess for idx, sess in enumerate(session[:-1]) 
                              if idx % chans_per_sess in params.free_chans
                              and np.shape(eventTimes[idx])[-1] > 1
                              and eventTimes[idx+len(params.free_chans)].shape[1] < 10]
            sess_to_drop = sess_to_drop + sess_to_drop_2
    
        if len(np.unique(session)) - len(sess_to_drop) > 1: 
            evCts = [np.shape(times)[-1] for times in eventTimes]
            only_good_sess = [sess for idx, (sess, ct) in enumerate(zip(session, evCts)) 
                              if idx % chans_per_sess in params.app_chans
                              and ct >= allExp_frameCounts[idx % chans_per_sess].shape[0]] 
        
            if len(only_good_sess) == 1:
                sess_to_drop = [sess for sess in np.unique(session) if sess not in only_good_sess]
    
        
        eventTimes = [times for sess, times in zip(session, eventTimes) if sess not in sess_to_drop]
        breakTimes = [times for sess, times in zip(session, breakTimes) if sess not in sess_to_drop]
        allExp_signalTimes = [times for sess, times in zip(session, allExp_signalTimes) if sess not in sess_to_drop]
        session = [sess for sess in session if sess not in sess_to_drop]

        
        uniqueSessions = np.unique(session)
        numSessions = len(uniqueSessions)
        chans_per_sess = int(len(allExp_signalTimes) / numSessions)
    
        # clean out weird extra signals
        nsx_counts = []
        for expNum, fCounts in enumerate(allExp_frameCounts):
            tmp_nsx_count = []
            for sess in range(numSessions):
                
                camTimes = allExp_signalTimes[int(sess*chans_per_sess + expNum)]
                evTimes = eventTimes[int(sess*chans_per_sess + expNum)]
                
                for eventNum in range(evTimes.shape[-1]):
                    start_time = evTimes[0, eventNum]
                    end_time = evTimes[1, eventNum]
                    
                    event_camTimes = camTimes[np.logical_and(camTimes >= start_time, camTimes <= end_time)]
                    tmp_nsx_count.append(event_camTimes.shape[0])
            nsx_counts.append(np.array(tmp_nsx_count))
        
        for idx, (nsxCount, fCounts) in enumerate(zip(nsx_counts, allExp_frameCounts)):
            if int(date) == 20210311 and tmpIdx == 0 and idx == 0:
                nsxCount[0] = -1000000
            diff = len(nsxCount) - fCounts.shape[0]
            if diff < 0:
                shifted_frameDiffs = np.array([abs(fCounts.cam1 - np.hstack((np.zeros((shift,))*np.nan, 
                                                                             nsxCount, 
                                                                             np.zeros((len(fCounts.cam1) - len(nsxCount) - shift,))*np.nan))).sum() 
                                               for shift in range(abs(diff)+1)])
                shift = np.argmin(shifted_frameDiffs)
                removed_eventTimes.append(np.empty((2, 0)))
                eventTimes[idx] = np.hstack((np.zeros((2, shift))*np.nan, 
                                             eventTimes[idx], 
                                             np.zeros((2, len(fCounts.cam1) - len(nsxCount) - shift))*np.nan))                 
            else:
                shifted_frameDiffs = np.array([abs(fCounts.cam1 - nsxCount[shift:len(nsxCount)- (diff - shift)]).sum() for shift in range(diff+1)])
                shift = np.argmin(shifted_frameDiffs)
                removed_eventTimes.append(np.hstack((eventTimes[idx][:, :shift], eventTimes[idx][:, len(nsxCount)- (diff - shift):])))  
                eventTimes[idx] = eventTimes[idx][:, shift:len(nsxCount)- (diff - shift)]

    allExp_eventInfo = []
    allExp_event_frameTimes = []
    allExp_bad_frameTimes = []
    allExp_breakInfo = []
    for expNum, (exp, fCounts) in enumerate(zip(expNames, allExp_frameCounts)):
        event_info = pd.DataFrame(np.empty((fCounts.shape[0], 5)), 
                                  columns = ['exp_name', 
                                             'ephys_session', 
                                             'video_session', 
                                             'start_time', 
                                             'end_time'])
        brExp = []
        brEphysSess = []
        brVidSess = []
        brEvent = []
        brFrameNum = []
        
        event_frameTimes = []
        badEvent_frameTimes = []
        evCount = 0
        for sess in range(numSessions):
            
            camTimes = allExp_signalTimes[int(sess*chans_per_sess + expNum)]
            evTimes = eventTimes[int(sess*chans_per_sess + expNum)]
            bad_evTimes = removed_eventTimes[int(sess*chans_per_sess + expNum)]
            
            if int(date) < 20210117:
                brTimes = breakTimes[int(sess*chans_per_sess + expNum)]
            else:
                brTimes = []
            
            for eventNum in range(evTimes.shape[-1]):
                start_time = evTimes[0, eventNum]
                end_time = evTimes[1, eventNum]
                
                event_camTimes = camTimes[np.logical_and(camTimes >= start_time, camTimes <= end_time)]
                fCounts.nsx_count[evCount] = event_camTimes.shape[0]
                
                cam_frame_counts = [ct for ct, colName in zip(fCounts.iloc[evCount, :], fCounts.columns) if 'cam' in colName]
                if eventNum+1 == evTimes.shape[-1] and fCounts.nsx_count[evCount] > max(cam_frame_counts):
                    event_camTimes = event_camTimes[:int(max(cam_frame_counts))]
                    fCounts.nsx_count[evCount] = event_camTimes.shape[0]
                    end_time = event_camTimes[int(max(cam_frame_counts))-1]
                
                event_frameTimes.append(event_camTimes)
                
                vPath = allExp_vidPaths[expNum][maxEvent_camNum[expNum]][eventNum]
                try: 
                    video_session = int(os.path.basename(vPath).split('_s')[1][0])
                except:
                    video_session = int(os.path.basename(vPath).split('_session')[1][0])
                                        
                event_info.iloc[evCount, :] = [exp, uniqueSessions[sess]+1, video_session, start_time, end_time]
                
                evCount += 1
            for eventNum in range(bad_evTimes.shape[-1]):
                start_time = bad_evTimes[0, eventNum]
                end_time   = bad_evTimes[1, eventNum]
                event_camTimes = camTimes[np.logical_and(camTimes >= start_time, camTimes <= end_time)]
                
                badEvent_frameTimes.append(event_camTimes)
            

            if len(brTimes) > 0:
                for brT in brTimes[0]:
                    brEv = [eventNum for eventNum in range(evTimes.shape[-1]) if brT > evTimes[0, eventNum] and brT < evTimes[1, eventNum]]
                    if len(brEv) > 0:
                        brEv = brEv[0]
                        breakFrame = np.sum(event_frameTimes[brEv] <= brT)
                        
                        brExp.append(exp)
                        brEphysSess.append(uniqueSessions[sess]+1)
                        brVidSess.append(video_session)
                        brEvent.append(brEv + 1)
                        brFrameNum.append(breakFrame)
        
        break_info = pd.DataFrame(zip(brExp, brEphysSess, brVidSess, brEvent, brFrameNum), 
                                  columns = ['exp_name', 
                                             'ephys_session', 
                                             'video_session', 
                                             'event', 
                                             'frame_before_break'])
                
                
        if int(date) == 20210301 and expNum == 1:
            event_info.iloc[2, :] = ['BeTL', 1, 1, tmp_start_time, tmp_end_time]
            event_frameTimes[2] = tmp_sigTimes
            allExp_frameCounts[1].iloc[2, 2] = len(tmp_sigTimes)
            
        allExp_eventInfo.append(event_info)
        allExp_event_frameTimes.append(event_frameTimes)
        allExp_bad_frameTimes.append(badEvent_frameTimes)        
        allExp_breakInfo.append(break_info)
    
    if int(date) >= 20210117: 
        
        # set up dict variables for saving to pickle and mat files
        saveData = {'event_info':         allExp_eventInfo, 
                    'frameTimes_byEvent': allExp_event_frameTimes, 
                    'frameCounts':        allExp_frameCounts, 
                    'experiments':        expNames,
                    'removed_frameTimes': allExp_bad_frameTimes,
                    'removed_eventTimes': removed_eventTimes}
        
        allExp_eventInfo_mat = np.empty((len(expNames),), dtype=np.object)
        allExp_frameCounts_mat = np.empty_like(allExp_eventInfo_mat)
        allExp_event_frameTimes_mat = np.empty_like(allExp_eventInfo_mat)
        expNames_mat = np.empty_like(allExp_eventInfo_mat)
        allExp_bad_frameTimes_mat = np.empty_like(allExp_eventInfo_mat)
        removed_eventTimes_mat = np.empty_like(allExp_eventInfo_mat) 
        for i in range(len(expNames)):
            
            
            allExp_eventInfo_mat[i]        = allExp_eventInfo[i].to_dict("list")
            allExp_frameCounts_mat[i]      = allExp_frameCounts[i].to_dict("list")
            removed_eventTimes_mat[i]      = removed_eventTimes[i]
            
            tmpTimes = allExp_event_frameTimes[i]
            tmpBadTimes = allExp_bad_frameTimes[i]
            eTimes = np.empty((len(tmpTimes),), dtype=np.object)
            bad_eTimes = np.empty((len(tmpBadTimes),), dtype=np.object)
            for eNum, times in enumerate(tmpTimes):
                eTimes[eNum] = times
            for eNum, badTimes in enumerate(tmpBadTimes):
                bad_eTimes[eNum] = badTimes
    
            allExp_event_frameTimes_mat[i] = eTimes
            allExp_bad_frameTimes_mat[i] = bad_eTimes     
            
            expNames_mat[i]                = expNames[i]
            
        saveData_mat = {'event_info':         allExp_eventInfo_mat, 
                        'frameTimes_byEvent': allExp_event_frameTimes_mat, 
                        'frameCounts':        allExp_frameCounts_mat, 
                        'experiments':        expNames_mat,
                        'removed_frameTimes': allExp_event_frameTimes_mat,
                        'removed_eventTimes': removed_eventTimes_mat}
    else:
        # set up dict variables for saving to pickle and mat files
        saveData = {'event_info'        : allExp_eventInfo, 
                    'frameTimes_byEvent': allExp_event_frameTimes, 
                    'frameCounts'       : allExp_frameCounts, 
                    'experiments'       : expNames,
                    'removed_frameTimes': allExp_bad_frameTimes,
                    'removed_eventTimes': removed_eventTimes,
                    'event_break_info'  : allExp_breakInfo}
        
        allExp_eventInfo_mat = np.empty((len(expNames),), dtype=np.object)
        allExp_frameCounts_mat = np.empty_like(allExp_eventInfo_mat)
        allExp_event_frameTimes_mat = np.empty_like(allExp_eventInfo_mat)
        expNames_mat = np.empty_like(allExp_eventInfo_mat)
        allExp_bad_frameTimes_mat = np.empty_like(allExp_eventInfo_mat)
        removed_eventTimes_mat = np.empty_like(allExp_eventInfo_mat) 
        allExp_breakInfo_mat = np.empty_like(allExp_eventInfo_mat)
        for i in range(len(expNames)):
            
            
            allExp_eventInfo_mat[i]        = allExp_eventInfo[i].to_dict("list")
            allExp_breakInfo_mat[i]        = allExp_breakInfo[i].to_dict("list")
            allExp_frameCounts_mat[i]      = allExp_frameCounts[i].to_dict("list")
            removed_eventTimes_mat[i]      = removed_eventTimes[i]
            
            tmpTimes = allExp_event_frameTimes[i]
            tmpBadTimes = allExp_bad_frameTimes[i]
            eTimes = np.empty((len(tmpTimes),), dtype=np.object)
            bad_eTimes = np.empty((len(tmpBadTimes),), dtype=np.object)
            for eNum, times in enumerate(tmpTimes):
                eTimes[eNum] = times
            for eNum, badTimes in enumerate(tmpBadTimes):
                bad_eTimes[eNum] = badTimes
    
            allExp_event_frameTimes_mat[i] = eTimes
            allExp_bad_frameTimes_mat[i] = bad_eTimes     
            
            expNames_mat[i]                = expNames[i]
            
        saveData_mat = {'event_info':         allExp_eventInfo_mat, 
                        'frameTimes_byEvent': allExp_event_frameTimes_mat, 
                        'frameCounts':        allExp_frameCounts_mat, 
                        'experiments':        expNames_mat,
                        'removed_frameTimes': allExp_event_frameTimes_mat,
                        'removed_eventTimes': removed_eventTimes_mat,
                        'event_break_info'  : allExp_breakInfo_mat}

    
    if operSystem == 'linux':
        writeMat    = os.path.join(path.tmpData, date + path.save_fileName + '.mat' )
        writePickle = os.path.join(path.tmpData, date + path.save_fileName + '.pkl' )
        with open(writePickle, 'wb') as fp:
            pickle.dump(saveData, fp, protocol = pickle.HIGHEST_PROTOCOL)
        savemat(writeMat, mdict = saveData_mat)
        subprocess.run(['sudo', 'mv', writePickle, os.path.join(path.analog_processed_dir, 'mat_files/')])
        subprocess.run(['sudo', 'mv', writeMat   , os.path.join(path.analog_processed_dir, 'pickle_files/')])
        
    elif operSystem == 'windows':   
        writeMat    = os.path.join(path.analog_processed_dir, 'mat_files',    date + path.save_fileName + '.mat')
        writePickle = os.path.join(path.analog_processed_dir, 'pickle_files', date + path.save_fileName + '.pkl')
        with open(writePickle, 'wb') as fp:
            pickle.dump(saveData, fp, protocol = pickle.HIGHEST_PROTOCOL)
        savemat(writeMat, mdict = saveData_mat)

#%% Check number of video frames for each video

data_paths = sorted(glob.glob(os.path.join(path.analog_processed_dir, 'pickle_files', '*')))

check_dates   = []
check_exp     = []
check_session = []
check_events  = []
check_counts  = []
for file in data_paths:
    date = os.path.basename(file)[:8]
    
    with open(file, 'rb') as fp:
        data = pickle.load(fp)
    
    for fCounts, exp, evInfo in zip(data['frameCounts'], data['experiments'], data['event_info']):    
        misMatches = np.where(np.sum(fCounts.eq(fCounts.iloc[:, 0], axis=0), axis = 1) < fCounts.shape[-1])[0]
        
        if exp == 'free':
            misMatches = np.array([idx for idx in misMatches if ~(fCounts.iloc[idx, 0] == fCounts.iloc[idx, 1] 
                                                                  and fCounts.iloc[idx, 0] >= fCounts.iloc[idx, 2])])
                        
                         
        if len(misMatches) > 0:
            for idx in misMatches:
                check_dates.append(date)
                check_exp.append(exp)
                check_session.append(int(evInfo.video_session[idx]))
                check_events.append(idx)
                check_counts.append(tuple(fCounts.iloc[idx, :]))

data_to_check = pd.DataFrame(zip(check_dates,
                                 check_exp,
                                 check_session,
                                 check_events,
                                 check_counts),
                             columns=['date',
                                      'experiment',
                                      'session',
                                      'event',
                                      'counts'])

procDate_start = os.path.basename(data_paths[0])[:8]
procDate_end   = os.path.basename(data_paths[-1])[:8]
writePickle = os.path.join(path.analog_processed_dir, procDate_start + '_to_' + procDate_end + '_data_to_check'  + '.pkl')
with open(writePickle, 'wb') as fp:
    pickle.dump(data_to_check, fp, protocol = pickle.HIGHEST_PROTOCOL)
        
    
#%%

idx = 3
file = data_paths[idx]

date = os.path.basename(file)[:8]
print(date)
with open(file, 'rb') as fp:
    data = pickle.load(fp)
                

#%%
    
#with open(path.storage + '2019_11_26_foraging_and_homeCage_spikeData.p', 'rb') as f:
#    spikeData = pickle.load(f)
#
#with open(path.storage + '2019_11_26_foraging_and_homeCage_analogSignals.p', 'rb') as f:
#    analogSignals = pickle.load(f)


#%% testing, delete after this

#chan1 = nsx.getdata(elec_ids = [129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 140], start_time_s= 10, data_time_s= 100)
##chan1 = nsx.getdata(elec_ids = 'all', start_time_s = 10, data_time_s = 100)
#signals = chan1['data']
##plt.plot(signals[0, 0:100000])
    