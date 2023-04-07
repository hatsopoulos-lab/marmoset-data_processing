#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 21:22:44 2023

@author: daltonm
"""

import numpy as np
import pandas as pd
import dill
import os
import glob
import re
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, behavior
from pynwb.epoch import TimeIntervals
from ndx_pose import PoseEstimationSeries, PoseEstimation
import datetime

anipose_base = '/project/nicho/data/marmosets/kinematics_videos/moths/TYJL/'

class dpath:
    base = [anipose_base]
    dates = ['2021_02_11']  # for now we can only do one date at a time
    reach_data = '/project/nicho/data/marmosets/processed_datasets/reach_and_trajectory_information/20210211_reach_and_trajectory_info_updated.pkl'
    nwb_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003_testing_pose.nwb'
    nwb_outfile = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003_testing_pose_NEW.nwb'
    session = 1
    expName = 'moths'

def load_dlc_data(data_dirs):
    data_files = []
    for data_dir in data_dirs:
        data_files.extend(sorted(glob.glob(os.path.join(data_dir, '*.csv'))))
        
    dlc = []
    dlc_metadata = []    
    video_event_info = pd.DataFrame(np.empty((len(data_files), 3)), columns = ['date', 'event', 'recording_marm'])
    for fNum, f in enumerate(data_files):
        data = pd.read_csv(f)
        dataIdx = sorted(list(range(0, data.shape[1]-13, 6)) + 
                         list(range(1, data.shape[1]-13, 6)) + 
                         list(range(2, data.shape[1]-13, 6))) 
        metadataIdx = sorted(list(range(3, data.shape[1]-13, 6)) + 
                              list(range(4, data.shape[1]-13, 6)) + 
                              list(range(5, data.shape[1]-13, 6)))
        dlc_tmp = data.iloc[:, dataIdx].to_numpy(dtype=np.float64)
        dlc_metadata_tmp = data.iloc[:, metadataIdx]
        
        dlc_tmp = dlc_tmp.T
        dlc_tmp = np.reshape(dlc_tmp, (int(dlc_tmp.shape[0]/3), 3, dlc_tmp.shape[1]))
        
        dlc.append(dlc_tmp)
        dlc_metadata.append(dlc_metadata_tmp)
        
        event_name = os.path.basename(f)
        namePattern = re.compile('^[a-zA-Z]{4}_[\d]*_[\d]*_[\d]*')
        
        video_event_info.iloc[fNum] = [re.findall(namePattern, event_name)[0], 
                                       int(event_name.split('event')[1][:3]), 
                                       False]
            
    return dlc, dlc_metadata, video_event_info 

def load_reach_data(datapath):
    with open(datapath, 'rb') as f:
        reach_data = dill.load(f)
    
    return reach_data
    

def filtered_kinematics_to_processing_module(event_data, meta, video_event_timeseries, series_name, original_videos, labeled_videos, kinematics_pm):
    pose_estimation_series = []                    
    for mIdx, mName in enumerate(event_data['marker_names']):
        position = event_data['position'][mIdx].T
        reproj_error = meta['%s_error' % mName].to_numpy()  # a confidence value for every frame
        marker_pose = PoseEstimationSeries(
            name=mName,
            description='3D pose tracked by DLC+Anipose. Dimensions of data are [time, x/y/z]',
            data=position,
            unit='m',
            conversion = 1e-2,
            reference_frame='(0,0,0) corresponds to the near left corner of the prey capture/foraging arena or touchscreen, viewed from the marmoset perspective\n' +
                            '+X-axis points to near right corner. +Y-axis points to far left corner. +Z-axis is up, orthogonal to arena floor.',
            timestamps=video_event_timeseries,
            confidence=reproj_error,
            confidence_definition='Reprojection error output from Anipose, in pixels.',
        )
    
        pose_estimation_series.append(marker_pose) 
    
    pe = PoseEstimation(
        pose_estimation_series=pose_estimation_series,
        name = series_name,
        description='3D positions of all markers using DLC+Anipose, with post-Anipose cleanup',
        original_videos=original_videos,
        labeled_videos=labeled_videos,
        dimensions=np.array([[1440, 1080], [1440, 1080]], dtype='uint16'),
        scorer=scorer,
        source_software='DeepLabCut+Anipose',
        source_software_version='2.2b8',
        nodes=event_data['marker_names'],
        #edges=np.array([[0, 1]], dtype='uint8'),
        # devices=[camera1, camera2],  # this is not yet supported
    )  
    
    kinematics_pm.add(pe)
        
def unfiltered_kinematics_to_processing_module():
    test = []

def original_dlc_before_anipose_to_processing_module():
    test = []
    
def reaching_segments_to_intervals_module(reach_seg_intervals, event_data, series_name, video_event_timeseries, kin_pm_name):
    event_times = video_event_timeseries.timestamps[:]
    for start_idx, stop_idx in zip(event_data['starts'], event_data['stops']):
        stop_idx = stop_idx - 1
        peak_idxs  = [idx for idx in event_data['peaks'] if idx > start_idx and idx < stop_idx]
        start_time = event_times[start_idx]
        stop_time  = event_times[stop_idx]
        peak_times = event_times[peak_idxs]
        
        peak_idxs_string = str()
        for idx in peak_idxs:
            peak_idxs_string += (str(idx) + ',')
        peak_idxs_string = peak_idxs_string[:-1]
        
        peak_times_string = str()
        for pkTime in peak_times:
            peak_times_string += (str(pkTime) + ',')
        peak_times_string = peak_times_string[:-1]
        
        reach_seg_intervals.add_row(start_time           = start_time, 
                                    stop_time            = stop_time,
                                    peak_extension_times = peak_times_string,
                                    start_idx            = start_idx,
                                    stop_idx             = stop_idx,
                                    peak_extension_idxs  = peak_idxs_string,
                                    kinematics_module    = kin_pm_name,
                                    video_event          = series_name)
                
def get_supplemental_info_for_modules(expName, session, event_data, nwb):
    # meta_list = []
    # unprocessed_pose_list = []
    # timestamps_base_key_list = []
    # timestamps_video_event_key_list = []
    # series_name_list = []
    # original_videos_list = []
    # labeled_videos_list = []
        
    try:
        event_num = event_data['episode']
        event_idx = video_event_info.index[video_event_info['episode'] == event_num][0]
    except:
        event_num = event_data['event']
        event_idx = video_event_info.index[video_event_info['event']   == event_num][0]
    
    meta = dlc_metadata[event_idx]
    unprocessed_pose = dlc[event_idx]
    
    timestamps_base_key = 'video_event_timestamps_%s' % expName
    timestamps_video_event_key = '%s_s_%d_e_%s_timestamps' % (expName, session, str(event_num).zfill(3))
    video_event_timeseries = nwb.processing[timestamps_base_key].data_interfaces[timestamps_video_event_key]            
    series_name = '%s_s_%d_e_%s_position' % (expName, session, str(event_num).zfill(3))
    try:
        original_videos = glob.glob(os.path.join(dpath.base[0], dpath.dates[0], 'avi_videos', '*s_%d_e_%s_*' % (session, str(event_num).zfill(3))))
        test = original_videos[0]
    except:
        original_videos = glob.glob(os.path.join(dpath.base[0], dpath.dates[0], 'avi_videos', '*session%d_event%s_*' % (session, str(event_num).zfill(3))))
    
    if 'videos-labeled-proj' in glob.glob(os.path.join(dpath.base[0], dpath.dates[0], '*')):
        labeled_dir = 'videos-labeled-proj'
    else:
        labeled_dir = 'videos-labeled-filtered'
        
    try:
        labeled_videos  = glob.glob(os.path.join(dpath.base[0], dpath.dates[0], labeled_dir, '*s_%d_e_%s_*' % (session, str(event_num).zfill(3))))
        test = labeled_videos[0]
    except:
        labeled_videos  = glob.glob(os.path.join(dpath.base[0], dpath.dates[0], labeled_dir, '*session%d_event%s_*' % (session, str(event_num).zfill(3))))
        
        # meta_list.append(meta)
        # unprocessed_pose_list.append(unprocessed_pose)
        # timestamps_base_key_list.append(timestamps_base_key)
        # timestamps_video_event_key_list.append(timestamps_video_event_key)
        # series_name_list.append(series_name)
        # original_videos_list.append(original_videos)
        # labeled_videos_list.append(labeled_videos)
        
    return meta, unprocessed_pose, video_event_timeseries, series_name, original_videos, labeled_videos

def add_modules_to_nwb(kin_pm_name, kin_pm_description, kin_intervals_name, kin_intervals_description, nwb):    
    # add processing module for experiment kinematics
    if kin_pm_name is not None:
        kinematics_pm = nwb.create_processing_module(name=kin_pm_name, description=kin_pm_description)
    else:
        print('Need to define a name for the kinematics processing module (this is probably a new experiment, or the expName passed to this function has an error).')
    
    # add intervals module linked to experiment kinematics. For goal_directed_kinematics, should be reaching_segments. For spontaneous, could be annotated behaviors such as grooming, eating, leaping, etc.
    if kin_intervals_name is not None:
        if 'reaching_segments' in kin_intervals_name:
            reach_seg_intervals = TimeIntervals(name = kin_intervals_name, description = kin_intervals_description)
            reach_seg_intervals.add_column(name="peak_extension_times", description="time of maximum extension (or 'peak' of reach)")
            reach_seg_intervals.add_column(name="start_idx"  , description="index of reach start within video event")
            reach_seg_intervals.add_column(name="stop_idx"   , description="index of reach stop within video event")
            reach_seg_intervals.add_column(name="peak_extension_idxs" , description="index of maximum extension (or 'peak' of reach) within video event")
            reach_seg_intervals.add_column(name="kinematics_module", description="string identifier for linked kinematics module. E.g. nwb.processing[kinematics_module].data_interfaces[video_event]")
            reach_seg_intervals.add_column(name="video_event", description="string identifier for video event in linked kinematics module. E.g. nwb.processing[kinematics_module].data_interfaces[video_event]")
        else:
            print('The structure of the intervals table has not been defined for kinematics intervals with the name %s' % kin_intervals_name)
    else:
        print('Need to define a name for the kinematic intervals module (this is probably a new experiment, or the expName passed to this function has an error).')
            
    return kinematics_pm, reach_seg_intervals 

def add_kinematic_data_to_nwb(reach_data, dlc, dlc_metadata, video_event_info, expName, session, scorer, nwb_file):
    
    
    if expName == 'free':
        kin_pm_description = 'spontaneous behavior in the home enclosure'
        kin_pm_name = 'spontaneous_kinematics'
        kin_intervals_name = None
    elif expName == 'foraging':
        kin_pm_description = 'foraging kinematics in the apparatus'
        kin_pm_name = 'goal_directed_kinematics'
        kin_intervals_name = 'reaching_segments_%s' % expName
        kin_intervals_description = 'reaching segments identified in kinematics tracked by DLC+anipose'
    elif expName == 'crickets' or expName == 'moths':
        kin_pm_description = 'prey capture kinematics - %s' % expName
        kin_pm_name = 'goal_directed_kinematics'
        kin_intervals_name = 'reaching_segments_%s' % expName
        kin_intervals_description = 'reaching segments identified in kinematics tracked by DLC+anipose'
    elif expName.lower() == 'betl':
        kin_pm_description = 'kinematics of virtual prey capture in BeTL task'
        kin_pm_name = 'goal_directed_kinematics'
        kin_intervals_name = 'reaching_segments_%s' % expName
        kin_intervals_description = 'reaching segments identified in kinematics tracked by DLC+anipose'
    else:
        kin_pm_description = 'kinematics for %s experiment' % expName
        kin_pm_name = None
        kin_intervals_name = None
        
    
    with NWBHDF5IO(nwb_file, 'r+') as io:
        nwb = io.read()
            
        kinematics_pm, reach_seg_intervals = add_modules_to_nwb(kin_pm_name, kin_pm_description, kin_intervals_name, kin_intervals_description, nwb)
        
        for reachIdx, event_data in enumerate(reach_data):
        
            meta, unprocessed_pose, video_event_timeseries, series_name, original_videos, labeled_videos = get_supplemental_info_for_modules(expName, session, event_data, nwb)     
        
            filtered_kinematics_to_processing_module(event_data, 
                                                     meta, 
                                                     video_event_timeseries, 
                                                     series_name, 
                                                     original_videos, 
                                                     labeled_videos,
                                                     kinematics_pm)
            
            
            

            # pose_estimation_series = []                    
            # for mIdx, mName in enumerate(event_data['marker_names']):
            #     position = event_data['position'][mIdx].T
            #     reproj_error = meta['%s_error' % mName].to_numpy()  # a confidence value for every frame
            #     marker_pose = PoseEstimationSeries(
            #         name=mName,
            #         description='3D pose tracked by DLC+Anipose. Dimensions of data are [time, x/y/z]',
            #         data=position,
            #         unit='m',
            #         conversion = 1e-2,
            #         reference_frame='(0,0,0) corresponds to the near left corner of the prey capture/foraging arena or touchscreen, viewed from the marmoset perspective\n' +
            #                         '+X-axis points to near right corner. +Y-axis points to far left corner. +Z-axis is up, orthogonal to arena floor.',
            #         timestamps=video_event_timeseries,
            #         confidence=reproj_error,
            #         confidence_definition='Reprojection error output from Anipose (in pixels)',
            #     )
            
            #     pose_estimation_series.append(marker_pose) 
            
            # pe = PoseEstimation(
            #     pose_estimation_series=pose_estimation_series,
            #     name = series_name,
            #     description='3D positions of all markers using DLC+Anipose, with post-Anipose cleanup',
            #     original_videos=original_videos,
            #     labeled_videos=labeled_videos,
            #     dimensions=np.array([[1440, 1080], [1440, 1080]], dtype='uint16'),
            #     scorer=scorer,
            #     source_software='DeepLabCut+Anipose',
            #     source_software_version='2.2b8',
            #     nodes=event_data['marker_names'],
            #     #edges=np.array([[0, 1]], dtype='uint8'),
            #     # devices=[camera1, camera2],  # this is not yet supported
            # )
            
            unfiltered_kinematics_to_processing_module()
            original_dlc_before_anipose_to_processing_module()
            
            reaching_segments_to_intervals_module(reach_seg_intervals, 
                                                  event_data, 
                                                  series_name, 
                                                  video_event_timeseries,
                                                  kin_pm_name)

        nwb.add_time_intervals(reach_seg_intervals)        
        
        # with NWBHDF5IO(dpath.nwb_outfile, mode='w') as export_io:
        #     export_io.export(src_io=io, nwbfile=nwb)
        
        io.write(nwb)

        
if __name__ == '__main__':
    reach_data = load_reach_data(dpath.reach_data)
    
    for base in dpath.base:
        data_dirs = []
        for date in dpath.dates:
            data_dirs.append(os.path.join(base, date, 'pose-3d'))
    
        dlc, dlc_metadata, video_event_info = load_dlc_data(data_dirs)   
    
    # with open(os.path.join(dpath.base[0], dpath.dates[0], 'scorer_info.txt')) as f:
    #     scorer = f.readlines()[0]    
    # scorer = scorer.split('filtered')[-1]
    # scorer = scorer.split('_meta')[0]
    scorer = 'DLC_resnet50_simple_joints_modelApr8-trainset95shuffle1_420000'
    
    add_kinematic_data_to_nwb(reach_data, dlc, dlc_metadata, video_event_info, dpath.expName, dpath.session, scorer= None, nwb_file=dpath.nwb_file)