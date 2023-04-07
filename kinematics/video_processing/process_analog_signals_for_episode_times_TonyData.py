#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:27:47 2020

@author: daltonm
"""

##### Need to test the two versions of mat files to see if they are providing the same information

import cv2
import numpy as np
import pandas as pd
from pandas import HDFStore
from brpylib import NevFile, NsxFile
import dill
import matplotlib.pyplot as plt
import shutil
import h5py
import subprocess
from scipy.io import savemat, loadmat
import os
import glob
import re
import time
from scipy.signal import savgol_filter
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, behavior
from pynwb.epoch import TimeIntervals
from ndx_pose import PoseEstimationSeries, PoseEstimation
import datetime
import argparse

class params:
    
    expDetector = 1
    camSignal_voltRange = [2900, 3000]
    downsample = 10
    eventDetectTime = .29
    eventDetector = eventDetectTime * 30000 / downsample
    break_detector = .06 * 30000 / downsample
    analogChans = [129, 130, 131]
    free_chans = [0]
    app_chans = [1]
    BeTL_chans = [2]
    num_app_cams = 2
    num_free_cams = 2
    nsx_filetype = 'ns6'

    minimum_free_session_minutes = 5

def get_filepaths(ephys_path, kin_path, marms_ephys_code, marms_kin_code, dates):

    dates = [date.replace('_', '') for date in dates]    

    ephys_folders = sorted(glob.glob(os.path.join(ephys_path, marms_ephys_code + '*')))
    ephys_folders = [fold for fold in ephys_folders 
                     if re.findall(datePattern, os.path.basename(fold))[0] in dates
                     and any(exp in os.path.basename(fold).lower() for exp in experiments)]    
    print(ephys_folders)
    kin_outer_folders = sorted(glob.glob(os.path.join(kin_path, '*')))
    kin_outer_folders = [fold for fold in kin_outer_folders if any(exp in os.path.basename(fold).lower() for exp in experiments)]
    kin_folders = []
    for outFold in kin_outer_folders:
        inner_folders = glob.glob(os.path.join(outFold, marms_kin_code, '*'))
        weird_folders = [fold for fold in inner_folders if '.toml' not in fold and len(os.path.basename(fold).replace('_', '')) > 8]
        if len(weird_folders) > 0:
            print('These are weird folders. They will be processed but you should take note of them in case you want to delete the processed data')
            print(weird_folders)
            
        inner_folders = [fold for fold in inner_folders if '.toml' not in fold and os.path.basename(fold).replace('_', '')[:8] in dates]
        inner_folders = [fold.replace('\\', '/') for fold in inner_folders]
        kin_folders.extend(inner_folders)
        
    return ephys_folders, kin_folders    

def store_drop_records(timestamps, dropframes_proc_mod, drop_record_folder, exp_name, sNum, epNum):
    
    camPattern = re.compile(r'cam\d{1}')
    drop_records = sorted(glob.glob(os.path.join(drop_record_folder, 
                                                 '*session%d*event_%s_droppedFrames.txt' % (sNum, epNum))))
    
    description = 'Boolean vector of good frames (True) and dropped/replaced frames (False) for given session/episode/camera.'
    if len(drop_records) > 0:
        for rec in drop_records:
            camNum = re.findall(camPattern, rec)[0]
            record_name = '%s_s_%d_e_%s_%s_dropFramesMask' % (exp_name, sNum, epNum, camNum)
            dropped_frames = np.loadtxt(drop_records[0], delimiter=',', dtype=str)
            dropped_frames = [int(framenum)-1 for framenum in dropped_frames[:-1]]
            data = np.full((len(timestamps),), True)
            data[dropped_frames] = False 
            if record_name not in dropframes_proc_mod.data_interfaces.keys():
                cam_drop_record = TimeSeries(name=record_name,
                                             data=data,
                                             unit="None",
                                             timestamps=timestamps,
                                             description = description,
                                             continuity = 'continuous'
                                             )
                dropframes_proc_mod.add(cam_drop_record)
    
    return
            
            
def timestamps_to_nwb(nwbfile_path, kin_folders, saveData, expNames):
    ###### TO NWB ######
    # open the NWB file in r+ mode    

    opened = False
    file_error = True
    while not opened:
        try:
            with NWBHDF5IO(nwbfile_path, 'r+') as io:
                nwbfile = io.read()
                
                file_error = False
                
                nwbfile.keywords = expNames
                
                # create a TimeSeries and add it to the processing module 'episode_timestamps_EXPNAME'
                sessPattern = re.compile('[0-9]{3}.nwb') 
                sessNum = int(re.findall(sessPattern, nwbfile_path)[-1][:3])
                for frame_times, event_info, exp_name in zip(saveData['frameTimes_byEvent'], saveData['event_info'], saveData['experiments']):

                    timestamps_module_name = 'video_event_timestamps_%s' % exp_name
                    timestamps_module_desc = 'set of timeseries holding timestamps for each behavior/video event for experiment = %s' % exp_name
    
                    if timestamps_module_name in nwbfile.processing.keys():
                        timestamps_proc_mod = nwbfile.processing[timestamps_module_name]
                    else:
                        timestamps_proc_mod = nwbfile.create_processing_module(name=timestamps_module_name,
                                                                               description=timestamps_module_desc)
                    
                    dropframes_module_name = 'dropped_frames_%s' % exp_name
                    dropframes_module_desc = ('Record of dropped frames. The dropped frames have ' +
                                              'been replaced by copies of the previous good frame. ' + 
                                              'Pose estimation may not be effected if most of the cameras ' +
                                              'captured that frame or if the drop is brief. Use the boolean ' +
                                              'mask vectors stored here as needed.')
                    if dropframes_module_name in nwbfile.processing.keys():
                        dropframes_proc_mod = nwbfile.processing[dropframes_module_name]
                    else:
                        dropframes_proc_mod = nwbfile.create_processing_module(name=dropframes_module_name,
                                                                               description=dropframes_module_desc)
                    
                    video_events_intervals_name = 'video_events_%s' % exp_name
                    if video_events_intervals_name in nwbfile.intervals.keys():
                        epi_mod_already_exists = True
                        video_events = nwbfile.intervals[video_events_intervals_name]
                        video_events_df = video_events.to_dataframe()                    
                    else:
                        epi_mod_already_exists = False
                        video_events = TimeIntervals(name = video_events_intervals_name,
                                                     description = 'metadata for behavior/video events associated with kinematics')
                        video_events.add_column(name="video_session", description="video session number of recorded video files")
                        video_events.add_column(name="analog_signals_cut_at_end", description="The number of analog signals (if any) that occurred after the end of video recording session. If they existed, they were cut during processing.")
                    

                    drop_record_folder = os.path.join([fold for fold in kin_folders if '/%s/' % exp_name in fold][0],
                                                      'drop_records')
                    for eventIdx, timestamps in enumerate(frame_times):
                        if event_info.ephys_session[eventIdx] == sessNum:
                            series_name = '%s_s_%d_e_%s_timestamps' % (event_info.exp_name[eventIdx], 
                                                                       int(event_info.video_session[eventIdx]), 
                                                                       str(int(event_info.episode_num[eventIdx])).zfill(3)) 
                                                
                            if series_name not in nwbfile.processing[timestamps_module_name].data_interfaces.keys(): 
                                data = np.full((len(timestamps), ), np.nan)
                                episode_timestamps = TimeSeries(name=series_name,
                                                                data=data,
                                                                unit="None",
                                                                timestamps=timestamps,
                                                                description = 'empty time series holding analog signal timestamps for video frames/DLC pose estimation that will be associated with PoseEstimationSeries data',
                                                                continuity = 'continuous'
                                                                )
                                
                                timestamps_proc_mod.add(episode_timestamps)
                            
                                if not epi_mod_already_exists or not any(video_events_df.start_time == event_info.start_time[eventIdx]):
                                    video_events.add_row(start_time                = event_info.start_time[eventIdx], 
                                                         stop_time                 = event_info.end_time[eventIdx], 
                                                         video_session             = event_info.video_session[eventIdx], 
                                                         analog_signals_cut_at_end = event_info.analog_signals_cut_at_end_of_session[eventIdx])
                                
                                store_drop_records(timestamps, 
                                                   dropframes_proc_mod,
                                                   drop_record_folder,
                                                   exp_name,
                                                   int(event_info.video_session[eventIdx]),
                                                   str(int(event_info.episode_num[eventIdx])).zfill(3))
                                
                    if video_events_intervals_name not in nwbfile.intervals.keys():
                        nwbfile.add_time_intervals(video_events) 

                io.write(nwbfile)        
            
            print('%s opened, edited, and written back to file. It is now closed.' % nwbfile_path)
            opened=True
        except:
            if file_error:
                print('%s is already open elsewhere. Waiting 10 seconds before trying again' % nwbfile_path)
                time.sleep(10)
            else:
                print('error occurred after file was loaded. Quitting')
                break



def remove_spurious_signals_and_sessions(eventTimes, session, chans_per_sess, allExp_signalTimes, breakTimes):       

    signals_to_drop = [idx for idx, times in enumerate(eventTimes) 
                        if np.shape(times)[-1] == 0 
                        or (times.shape[-1] <= 2 and np.all(np.diff(times, axis=0) < params.minimum_free_session_minutes*60))]
        
    sess_to_drop = [sess for idx, sess in enumerate(session) 
                    if idx in signals_to_drop 
                    and idx % chans_per_sess in params.free_chans]
    
    if len(np.unique(session)) - len(sess_to_drop) > 1:
        sess_to_drop_2 = [sess for idx, sess in enumerate(session[:-1]) 
                          if idx % chans_per_sess in params.free_chans
                          and np.shape(eventTimes[idx])[-1] > 1
                          and eventTimes[idx+(params.app_chans[0]-params.free_chans[0])].shape[1] < 5]
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
    
    return eventTimes, breakTimes, allExp_signalTimes, session, uniqueSessions, numSessions, chans_per_sess

def clean_remaining_spurious_signals(eventTimes, allExp_signalTimes, allExp_frameCounts, numSessions, chans_per_sess, removed_eventTimes, free_idx):

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
    
    for expNum, (nsxCount, fCounts) in enumerate(zip(nsx_counts, allExp_frameCounts)):
        diff = len(nsxCount) - fCounts.shape[0]
        if diff < 0:
            shifted_frameDiffs = np.array([abs(fCounts.cam1 - np.hstack((np.zeros((shift,))*np.nan, 
                                                                          nsxCount, 
                                                                          np.zeros((len(fCounts.cam1) - len(nsxCount) - shift,))*np.nan))).sum() 
                                           for shift in range(abs(diff)+1)])
            shift = np.argmin(shifted_frameDiffs)
            removed_eventTimes.append(np.empty((2, 0)))
            eventTimes[expNum] = np.hstack((np.zeros((2, shift))*np.nan, 
                                          eventTimes[expNum], 
                                          np.zeros((2, len(fCounts.cam1) - len(nsxCount) - shift))*np.nan)) 
        else:
            idx_adjust = 0
            remove_signal_idxs = []
            for ctIdx, fCt in enumerate(fCounts.cam1):
                if expNum == free_idx:
                    mismatch_thresh = 0.2 * fCt
                else:
                    mismatch_thresh = 10

                adjIdx = ctIdx + idx_adjust
                if abs(nsxCount[adjIdx] - fCt) > mismatch_thresh:
                    try:
                        shift = np.where(       abs(fCt - nsxCount[max(0, adjIdx-10):adjIdx+10]) == 
                                         np.min(abs(fCt - nsxCount[max(0, adjIdx-10):adjIdx+10])))[0][0] - (adjIdx - max(0, adjIdx-10)) 
                        
                        if shift > 0: 
                            remove_signal_idxs.extend(range(adjIdx, adjIdx+shift))
                        elif shift < 0:
                            remove_signal_idxs.append(range(adjIdx+shift, adjIdx))
                        idx_adjust += shift
                    except:
                        print('exception raised in spurious signal code for expNum %d and ctIdx %d' % (expNum, ctIdx))
                        continue
                elif ctIdx + 1 == len(fCounts.cam1):
                    remove_signal_idxs.extend(range(adjIdx+1, len(nsxCount)))
            
            keep_signal_idxs = [i for i in range(len(nsxCount)) if i not in remove_signal_idxs]
            keep_signal_idxs = keep_signal_idxs[ : fCounts.shape[0]]
            
            print('length of event times = %d, chans_per_sess = %d' % (len(eventTimes), chans_per_sess))
            sessEventCounters = [0]*int(len(eventTimes) / chans_per_sess)
            for chanIdx, evTimes in enumerate(eventTimes):
                if chanIdx % chans_per_sess == expNum:
                    sessNum = int(np.floor(chanIdx / chans_per_sess))
                    print(sessNum, evTimes.shape, len(sessEventCounters), sessEventCounters)
                    tmp_keep_idxs   = [i - sessEventCounters[sessNum] for i in keep_signal_idxs   
                                       if i >= sessEventCounters[sessNum] and i < evTimes.shape[-1] + sessEventCounters[sessNum]]
                    
                    tmp_remove_idxs = [i - sessEventCounters[sessNum] for i in remove_signal_idxs 
                                       if i >= sessEventCounters[sessNum] and i < evTimes.shape[-1] + sessEventCounters[sessNum]]
                    
                    removed_eventTimes.append(eventTimes[chanIdx][:, tmp_remove_idxs])
                    eventTimes[chanIdx] = eventTimes[chanIdx][:, tmp_keep_idxs] 
                    
                    sessEventCounters[sessNum] += evTimes.shape[-1]
            # expEventCounters = [0]*int(len(eventTimes) / chans_per_sess)
            # for chanIdx, evTimes in enumerate(eventTimes):
            #     if chanIdx % chans_per_sess == expNum:
            #         print(expNum, evTimes.shape, len(expEventCounters), expEventCounters)
            #         tmp_keep_idxs   = [i - expEventCounters[expNum] for i in keep_signal_idxs   
            #                            if i >= expEventCounters[expNum] and i < evTimes.shape[-1] + expEventCounters[expNum]]
                    
            #         tmp_remove_idxs = [i - expEventCounters[expNum] for i in remove_signal_idxs 
            #                            if i >= expEventCounters[expNum] and i < evTimes.shape[-1] + expEventCounters[expNum]]
                    
            #         removed_eventTimes.append(eventTimes[chanIdx][:, tmp_remove_idxs])
            #         eventTimes[chanIdx] = eventTimes[chanIdx][:, tmp_keep_idxs] 
                    
            #         expEventCounters[expNum] += evTimes.shape[-1]
    
    return eventTimes, removed_eventTimes 

def prepare_final_data_and_metadata(expNames, allExp_frameCounts, allExp_signalTimes, allExp_vidPaths, 
                                    eventTimes, removed_eventTimes, numSessions, chans_per_sess, kinFolders):
    allExp_eventInfo = []
    allExp_event_frameTimes = []
    allExp_bad_frameTimes = []
    allExp_breakInfo = []
    for expNum, (exp, fCounts) in enumerate(zip(expNames, allExp_frameCounts)):
        event_info = pd.DataFrame(np.empty((fCounts.shape[0], 7)), 
                                  columns = ['exp_name', 
                                             'ephys_session', 
                                             'video_session',
                                             'episode_num',
                                             'start_time', 
                                             'end_time',
                                             'analog_signals_cut_at_end_of_session'])
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
                        
            for eventNum in range(evTimes.shape[-1]):
                start_time = evTimes[0, eventNum]
                end_time = evTimes[1, eventNum]
                
                event_camTimes = camTimes[np.logical_and(camTimes >= start_time, camTimes <= end_time)]
                fCounts.nsx_count[evCount] = event_camTimes.shape[0]
                
                max_frames = int(np.max(fCounts.iloc[evCount, :-1]))
                analog_cut = fCounts.nsx_count[evCount] - max_frames
                if analog_cut > 0:
                    fCounts.nsx_count[evCount] = np.max(fCounts.iloc[evCount, :-1])
                    event_camTimes = event_camTimes[ : max_frames]  
                    end_time = event_camTimes[-1]
                
                event_frameTimes.append(event_camTimes)
                
                vPath = allExp_vidPaths[expNum][maxEvent_camNum[expNum]][evCount]
                try: 
                    video_session = int(os.path.basename(vPath).split('_s')[1][0])
                except:
                    video_session = int(os.path.basename(vPath).split('_session')[1][0])
                
                event_info.iloc[evCount, :] = [exp, 
                                               uniqueSessions[sess], 
                                               video_session, 
                                               eventNum+1, 
                                               start_time, 
                                               end_time, 
                                               analog_cut]
                
                evCount += 1
            for eventNum in range(bad_evTimes.shape[-1]):
                start_time = bad_evTimes[0, eventNum]
                end_time   = bad_evTimes[1, eventNum]
                event_camTimes = camTimes[np.logical_and(camTimes >= start_time, camTimes <= end_time)]
                
                badEvent_frameTimes.append(event_camTimes)
        
        break_info = pd.DataFrame(zip(brExp, brEphysSess, brVidSess, brEvent, brFrameNum), 
                                  columns = ['exp_name', 
                                             'ephys_session', 
                                             'video_session', 
                                             'event', 
                                             'frame_before_break'])
                           
        allExp_eventInfo.append(event_info)
        allExp_event_frameTimes.append(event_frameTimes)
        allExp_bad_frameTimes.append(badEvent_frameTimes)        
        allExp_breakInfo.append(break_info)
    
    # set up dict variables for saving to pickle and mat files
    saveData = {'event_info'        : allExp_eventInfo, 
                'frameTimes_byEvent': allExp_event_frameTimes, 
                'frameCounts'       : allExp_frameCounts, 
                'experiments'       : expNames,
                'removed_frameTimes': allExp_bad_frameTimes,
                'removed_eventTimes': removed_eventTimes,
                'event_break_info'  : allExp_breakInfo}
    
    save_fileName = '_experiment_event_and_frame_time_info'
    for kFold in kinFolders:
        metadata_path = os.path.join(kFold, 'metadata_from_kinematics_processing')
        os.makedirs(metadata_path, exist_ok=True)
        writePickle = os.path.join(metadata_path, date + save_fileName + '.pkl' )
        if not os.path.exists(writePickle):
            with open(writePickle, 'wb') as fp:
                dill.dump(saveData, fp, recurse=True)
    
    return saveData

def convert_saveData_to_matlab_compatible(saveData, expNames, allExp_frameCounts, removed_eventTimes, kinFolders):
    allExp_eventInfo_mat = np.empty((len(expNames),), dtype=np.object)
    allExp_frameCounts_mat = np.empty_like(allExp_eventInfo_mat)
    allExp_event_frameTimes_mat = np.empty_like(allExp_eventInfo_mat)
    expNames_mat = np.empty_like(allExp_eventInfo_mat)
    allExp_bad_frameTimes_mat = np.empty_like(allExp_eventInfo_mat)
    removed_eventTimes_mat = np.empty_like(allExp_eventInfo_mat) 
    allExp_breakInfo_mat = np.empty_like(allExp_eventInfo_mat)
    for i in range(len(expNames)):
        
        
        allExp_eventInfo_mat[i]        = saveData['event_info'][i].to_dict("list")
        allExp_breakInfo_mat[i]        = saveData['event_break_info'][i].to_dict("list")
        allExp_frameCounts_mat[i]      = allExp_frameCounts[i].to_dict("list")
        removed_eventTimes_mat[i]      = removed_eventTimes[i]
        
        tmpTimes    = saveData['frameTimes_byEvent'][i]
        tmpBadTimes = saveData['removed_frameTimes'][i]
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
    
    save_fileName = '_experiment_event_and_frame_time_info'
    for kFold in kinFolders:
        writeMat    = os.path.join(kFold, 'metadata_from_kinematics_processing', date + save_fileName + '.mat' )
        if not os.path.exists(writeMat):
            savemat(writeMat, mdict = saveData_mat)
    
    return saveData_mat

def get_video_frame_counts(matched_kinFolders, expNames):
    allExp_frameCounts = []
    allExp_vidPaths = []
    maxEvent_camNum = []
    for kFold, exp in zip(matched_kinFolders, expNames):
        if 'free' in exp.lower():
            num_cams = params.num_free_cams
        else:
            num_cams = params.num_app_cams
        
        vidPaths = []
        colNames = []
        for cNum in range(num_cams):
            tmp_vidPaths = glob.glob(os.path.join(kFold, 'avi_videos', f'*cam{cNum+1}*.avi'))           
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
    
    return allExp_vidPaths, allExp_frameCounts, maxEvent_camNum

def get_analog_frame_counts_and_timestamps(eFold, touchscreen = False, triggerData = None):
    analogFiles = sorted(glob.glob(os.path.join(eFold, '*.%s' % params.nsx_filetype)))
    allExp_signalTimes = []
    eventTimes = []
    breakTimes = []
    session = []
    for fNum, f in enumerate(analogFiles):
        nsx = NsxFile(f)
        analogChans = nsx.getdata(elec_ids = params.analogChans, downsample = params.downsample)
        try:
            signals = analogChans['data']

            if nsx_filetype == 'ns6':
                sampleRate = int(30000 / analogChans['downsample'])
            else:
                sampleRate = analogChans['samp_per_s']            
            
            try:
                times = np.linspace(0, analogChans['data_time_s'], num = int(analogChans['data_time_s'] * sampleRate))
            except:
                times = np.linspace(0, signals.shape[1] / sampleRate, signals.shape[1])
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
                    allExp_signalTimes.append(times[signalSamplesTmp])        
                    
                    if touchscreen and expChan in params.BeTL_chans:
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
                            if event_endSamples[-1] == len(times):
                                event_endSamples[-1] -= 1                       
                        else:
                            event_startSamples = expOpen_samples[0]
                            event_endSamples = expOpen_samples[-1]
                            if event_endSamples == len(times):
                                event_endSamples -= 1
                        
                    eventBoundariesTmp = np.vstack((event_startSamples, event_endSamples))
                    eventTimes.append(times[eventBoundariesTmp])
                    
                try:
                    sessPattern = re.compile('[0-9]{3}.ns') 
                    sessNum = int(re.findall(sessPattern, f)[-1][:-3])
                    session.append(sessNum)
                except:
                    print('ePhys sessionNum not present in filename. Saving fNum+1 as sessionNum')
                    session.append(fNum+1)
        except:
            try:
                sessPattern = re.compile('[0-9]{3}.ns') 
                sessNum = int(re.findall(sessPattern, f)[-1][:-3])
                session.append(sessNum)
            except:
                print('ePhys sessionNum not present in filename. Saving fNum+1 as sessionNum')
                session.append(fNum+1)
            eventTimes.append(np.array([]))
            breakTimes.append(np.array([]))
            allExp_signalTimes.append(np.array([]))        
            
        nsx.close()
    
    numSessions = len(np.unique(session))
    chans_per_sess = int(len(allExp_signalTimes) / numSessions)    

    return analogFiles, allExp_signalTimes, eventTimes, breakTimes, session, numSessions, chans_per_sess

def load_touchscreen_data(touchscreen_path, date):
    with open(os.path.join(touchscreen_path, 'touchscreen_data_%s.pkl' % date), 'rb') as f:
        ts_trialData = dill.load(f)
    return ts_trialData 

def reorder_signals_in_lists(allExp_frameCounts, allExp_vidPaths, expNames, maxEvent_camNum, eventTimes):
    numEvents = np.array([fCounts.shape[0] for fCounts in allExp_frameCounts])
    print(np.where(np.logical_or(numEvents == 1, numEvents == np.min(numEvents)))[0])
    free_idx = int(np.where(np.logical_or(numEvents == 1, numEvents == np.min(numEvents)))[0])
    if free_idx != params.free_chans[0]:
        fIdx = params.free_chans[0]
        allExp_frameCounts[fIdx], allExp_frameCounts[free_idx] = allExp_frameCounts[free_idx], allExp_frameCounts[fIdx] 
        allExp_vidPaths[fIdx], allExp_vidPaths[free_idx] = allExp_vidPaths[free_idx], allExp_vidPaths[fIdx] 
        expNames[fIdx], expNames[free_idx] = expNames[free_idx], expNames[fIdx]
        maxEvent_camNum[fIdx], maxEvent_camNum[free_idx] = maxEvent_camNum[free_idx], maxEvent_camNum[fIdx]
        eventTimes[fIdx], eventTimes[free_idx] = eventTimes[free_idx], eventTimes[fIdx]

        free_idx = fIdx
    
    return allExp_frameCounts, allExp_vidPaths, expNames, maxEvent_camNum, eventTimes, free_idx

def wait_for_all_batch_jobs_to_finish_video_conversion(matched_kinFolders):
    
    time_to_wait = 60*0.5
    stopwatch    = 0
    video_paths = [os.path.join(kFold, 'avi_videos') for kFold in matched_kinFolders]
    while not all([os.path.exists(vPath) for vPath in video_paths]):
        print('At least one experiment does not yet have the avi_videos path. Looping for another %f minutes. Time elapsed = %f minutes' % (time_to_wait / 60, stopwatch/60), flush=True)
        time.sleep(time_to_wait)
        stopwatch += time_to_wait

    time.sleep(time_to_wait)
    print('internal_check_1', flush=True)
    videos_done  = [False]*len(matched_kinFolders)
    previous_sum =     [0]*len(matched_kinFolders)
    updated_sum  =     [0]*len(matched_kinFolders) 
    stopwatch    = 0    
    while not all(videos_done):
        for idx, kFold in enumerate(matched_kinFolders): 
            print(idx, kFold, videos_done[idx], stopwatch)
            video_path = os.path.join(kFold, 'avi_videos')           
            
            updated_sum [idx] = sum(os.path.getsize('%s/%s' % (video_path, f)) for f in os.listdir('%s/.' % video_path))
            videos_done [idx] = (updated_sum[idx] == previous_sum[idx])
            previous_sum[idx] = updated_sum[idx]
        
        if stopwatch>time_to_wait*2:
            print('At least one experiment is still adding videos to avi_videos. Looping for another %f minutes. Time elapsed = %f minutes' % (time_to_wait / 60, stopwatch/60), flush=True)
        time.sleep(time_to_wait)
        stopwatch += time_to_wait
        
    print('\nAll videos completed. Moving on.\n')
    
    return

def convert_string_inputs_to_int_float_or_bool(orig_var):
    if type(orig_var) == str:
        orig_var = [orig_var]
    
    converted_var = []
    for v in orig_var:
        v = v.lower()
        try:
            v = int(v)
        except:
            pass
        try:
            v = float(v)
        except:
            v = None  if v == 'none'  else v
            v = True  if v == 'true'  else v
            v = False if v == 'false' else v 
        converted_var.append(v)
    
    if len(converted_var) == 1:
        converted_var = converted_var[0]
            
    return converted_var

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()

    # ap.add_argument("-v", "--vid_dir", required=True, type=str,
    #     help="path to directory holding kinematic data. E.g. /project/nicho/data/marmosets/kinematics_videos")
    # ap.add_argument("-ep", "--ephys_path", required=True, type=str,
    #     help="path to directory holding ephys data. E.g. /project/nicho/data/marmosets/electrophys_data_for_processing")
    # ap.add_argument("-m", "--marms", required=True, type=str,
    #  	help="marmoset 4-digit code, e.g. 'JLTY'")
    # ap.add_argument("-me", "--marms_ephys", required=True, type=str,
    #  	help="marmoset 2-digit code for ephys data, e.g. 'TY'")
    # ap.add_argument("-d", "--dates", nargs='+', required=True, type=str,
    #  	help="date(s) of recording (can have multiple entries separated by spaces)")
    # ap.add_argument("-e", "--exp_name", required=True, type=str,
    #  	help="experiment name, e.g. free, foraging, BeTL, crickets, moths, etc")
    # ap.add_argument("-e2", "--other_exp_name", required=True, type=str,
    #  	help="experiment name, e.g. free, foraging, BeTL, crickets, moths, etc")    
    # ap.add_argument("-t", "--touchscreen", required=True, type=str,
    #  	help="True or False to indicate whether touchscreen was used for experiment")    
    # ap.add_argument("-tp", "--touchscreen_path", required=True, type=str,
    #  	help="path to directory holding kinematic data")
    # ap.add_argument("-np", "--neur_proc_path", required=True, type=str,
    #  	help="path to directory holding neural processing code")
    # ap.add_argument("-meta", "--meta_path", required=True, type=str,
    #     help="path to metadata yml file to be added to NWB file, e.g. /project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/marms_complete_metadata.yml")
    # ap.add_argument("-prb", "--prb_path" , required=True, type=str,
    #     help="path to .prb file that provides probe/channel info to NWB file, e.g. /project/nicho/data/marmosets/prbfiles/MG_array.prb")
    # args = vars(ap.parse_args())

    args = {'vid_dir'        : '/project/nicho/data/marmosets/kinematics_videos',
            'ephys_path'     : '/project/nicho/data/marmosets/electrophys_data_for_processing',
            'marms'          : 'TYJL',
            'marms_ephys'    : 'TY',
            'dates'          : ['2021_02_11'],
            'exp_name'       : 'moths',
            'other_exp_name' : 'free',
            'touchscreen'    : 'False',
            'touchscreen_path' : 'BLANK',
            'neur_proc_path'   : '/project/nicho/projects/marmosets/code_database/data_processing/neural',
            'meta_path'        : '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/TY_complete_metadata.yml',
            'prb_path'         : '/project/nicho/data/marmosets/prbfiles/TY_02.prb',
            'debugging'        : True}

    
    touchscreen = convert_string_inputs_to_int_float_or_bool(args['touchscreen'])
    
    try:
        task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    except:
        task_id = 0
            
    if task_id == 0:    

        print('\n\n Beginning process_analog_signals_for_episode_times.py at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
        
        datePattern = re.compile('[0-9]{8}')         
        nsx_filetype = 'ns6'
    
        experiments = [args['exp_name'], args['other_exp_name']]
        print(args['ephys_path'])
        ephys_folders, kin_folders = get_filepaths(args['ephys_path'], args['vid_dir'], args['marms_ephys'], args['marms'], args['dates'])    
        print(ephys_folders)
        for eFold in ephys_folders:
            date = re.findall(datePattern, os.path.basename(eFold))[0]
            print(f'working on {date}')
            matched_kinFolders = [kFold for kFold in kin_folders if os.path.basename(kFold)[:10].replace('_', '') == date]
            expNames = [kinFold.split('/')[-3] for kinFold in matched_kinFolders]
        
            print(expNames)
            print(matched_kinFolders)
            wait_for_all_batch_jobs_to_finish_video_conversion(matched_kinFolders)
        
            if touchscreen:
                ts_trialData = load_touchscreen_data(args['touchscreen_path'], date)
            allExp_vidPaths, allExp_frameCounts, maxEvent_camNum = get_video_frame_counts(matched_kinFolders, expNames)
            analogFiles, allExp_signalTimes, eventTimes, breakTimes, session, numSessions, chans_per_sess = get_analog_frame_counts_and_timestamps(eFold)
        
            allExp_frameCounts, allExp_vidPaths, expNames, maxEvent_camNum, eventTimes, free_idx = reorder_signals_in_lists(allExp_frameCounts, 
                                                                                                                            allExp_vidPaths, 
                                                                                                                            expNames, 
                                                                                                                            maxEvent_camNum,
                                                                                                                            eventTimes)        
            
            removed_eventTimes = []
            for tmpIdx in range(2):
                eventTimes, breakTimes, allExp_signalTimes, session, uniqueSessions, numSessions, chans_per_sess = remove_spurious_signals_and_sessions(eventTimes, session, chans_per_sess, allExp_signalTimes, breakTimes)       
                eventTimes, removed_eventTimes = clean_remaining_spurious_signals(eventTimes, allExp_signalTimes, 
                                                                                  allExp_frameCounts, numSessions, 
                                                                                  chans_per_sess, removed_eventTimes, 
                                                                                  free_idx)
        
            saveData = prepare_final_data_and_metadata(expNames, allExp_frameCounts, allExp_signalTimes, allExp_vidPaths, 
                                                        eventTimes, removed_eventTimes, numSessions, chans_per_sess, matched_kinFolders) 
            # saveData_mat = convert_saveData_to_matlab_compatible(saveData, expNames, allExp_frameCounts, removed_eventTimes, matched_kinFolders)            
            for nsx_path in analogFiles:
                nwbfile_path = nsx_path.replace('.ns6', '.nwb')
                subprocess.call(['python',
                                  os.path.join(args['neur_proc_path'], 'store_neural_data_in_nwb.py'),
                                  '-f',  nsx_path,
                                  '-m',  args['meta_path'],
                                  '-p',  args['prb_path'],
                                  '-ab', 'yes'])
                timestamps_to_nwb(nwbfile_path, kin_folders, saveData, expNames)
                
        print('\n\n Ended process_analog_signals_for_episode_times.py at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
        