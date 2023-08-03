#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:34:05 2023

@author: daltonm
"""

# import toolboxes
import numpy as np
import re
import pynwb
from pynwb import NWBHDF5IO, TimeSeries
from pynwb.epoch import TimeIntervals
import glob
import datetime
import pytz
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time
import elephant

from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe, plot_probe_group

def read_prb_hatlab(file):
    """
    Read a PRB file and return a ProbeGroup object.
    Since PRB does not handle contact shapes, contacts are set to be circle of 5um radius.
    Same for the probe shape, where an auto shape is created.
    PRB format does not contain any information about the channel of the probe
    Only the channel index on device is given.
    Parameters
    ----------
    file : Path or str
        The file path
    Returns
    --------
    probegroup : ProbeGroup object
    """

    file = Path(file).absolute()
    assert file.is_file()
    with file.open("r") as f:
        contents = f.read()
    contents = re.sub(r"range\(([\d,]*)\)", r"list(range(\1))", contents)
    prb = {}
    exec(contents, None, prb)
    prb = {k.lower(): v for (k, v) in prb.items()}

    if "channel_groups" not in prb:
        raise ValueError("This file is not a standard PRB file")

    probegroup = ProbeGroup()
    imp = []
    for i, group in prb["channel_groups"].items():
        ndim = 2
        probe = Probe(ndim=ndim, si_units="um")

        try:
            chans = np.array(group["channels"], dtype="int64")
        except:
            chans = np.array(group["channels"], dtype=str)
            
        try:
            positions = np.array([group["geometry"][c] for c in chans], dtype="float64")
        except:
            positions = np.array([group["geometry"][idx] for idx, c in enumerate(chans)], dtype="float64")

        # try:
        #     chan_labels = np.array([group["chanels"][c] for c in chans], dtype="float64")
        # except:
        #     chan_labels = np.array([group["chan_label"][idx] for idx, c in enumerate(chans)], dtype="float64")        
        
        num_contacts = positions.shape[0]
        plane_axes = np.zeros((num_contacts, 2, ndim))
        plane_axes[:, 0, 0] = 1
        plane_axes[:, 1, 1] = 1
        
        probe.set_contacts(
            positions=positions, shapes="circle", shape_params={"radius": prb['radius']}, shank_ids=chans, plane_axes = plane_axes
        )
        # probe.create_auto_shape(probe_type="tip")

        probegroup.add_probe(probe)
        
        imp.append(np.array(group['impedance'][0], dtype=str))

    return probegroup, imp

def plot_prb(probegroup):
    probegroup_df = probegroup.to_dataframe()
    probenum = list(probegroup_df['shank_ids'])
    probenum = [str(prb).split('elec')[-1] for prb in probenum]
    plot_probe_group(probegroup, same_axes=True, with_channel_index=False)
    ax = plt.gca()
    for idx, prb in enumerate(probenum):
        try:
            ax.text(probegroup_df['x'][idx], probegroup_df['y'][idx], probegroup_df['z'][idx], prb)
        except:
            ax.text(probegroup_df['x'][idx], probegroup_df['y'][idx], prb)

    plt.show()

def create_nwb_copy_without_acquisition(nwb_infile, nwb_outfile):
    with NWBHDF5IO(nwb_infile, 'r') as io:
        nwb = io.read()    
        nwb.acquisition.clear()
        # video_timestamp_keys = [key for key in nwb.processing.keys() if 'video_event_timestamps' in key]
        # for key in video_timestamp_keys:
        #     nwb.processing.pop(key)
        try:
            mod_key = 'ecephys'
            nev_keys = [key for key in nwb.processing[mod_key].data_interfaces.keys() if 'nevfile' in key]
            for key in nev_keys:
                nwb.processing[mod_key].data_interfaces.pop(key)
        except:
            print('"%s" does not exist in the processing module' %mod_key)
        
        with NWBHDF5IO(nwb_outfile, mode='w') as export_io:
            export_io.export(src_io=io, nwbfile=nwb)  

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

def remove_duplicate_spikes_from_good_single_units(units, plot=False):
    for unitID in units.index:
        unit = units.loc[units.index == unitID, :]
        spike_times = unit.spike_times.iloc[0]

        thresh = 150
        if unit.quality.iloc[0] == 'good':
            unit_isi = elephant.statistics.isi(spike_times)
            tiny_isi =  np.where(unit_isi < thresh * 1e-6)[0]
            non_duplicate_idxs = np.setdiff1d(np.arange(spike_times.shape[0]), tiny_isi)
            spike_times = spike_times[non_duplicate_idxs]
            corrected_isi = elephant.statistics.isi(spike_times)
            
            print('unitID = %d, nSpikes_removed = %d' % (unitID, len(tiny_isi)))
        
            if len(tiny_isi) > 0:
                tmp = units.loc[units.index==unitID, 'spike_times'] 
                tmp.iloc[0] = spike_times
                # unit.iloc[0] = np.array([0, 1, 2])
                units.loc[units.index==unitID, 'spike_times'] = tmp
    
                if plot:
                    fig, (ax0, ax1) = plt.subplots(2, 1)
                    ax0.hist(corrected_isi, bins = np.arange(0, 0.05, 0.0005))
                    ax0.set_title('Corrected ISI (UnitID = %d)' % unitID)
                    ax1.hist(unit_isi, bins = np.arange(0, 0.05, 0.0005))
                    ax1.set_title('Original ISI')
                    ax1.set_xlabel('Seconds')
                    plt.show()
    
    return units

def get_sorted_units_and_apparatus_kinematics_with_metadata(nwb_prc, reaches_key, plot=False):
    units          = nwb_prc.units.to_dataframe()
    units          = remove_duplicate_spikes_from_good_single_units(units, plot=plot)
    reaches        = nwb_prc.intervals[reaches_key].to_dataframe()
    
    kin_module_key = reaches.iloc[0].kinematics_module
    kin_module = nwb_prc.processing[kin_module_key]
    
    return units, reaches, kin_module

def get_raw_timestamps(nwb_acq):
    # create timestamps for raw neural data from starting_time, rate, and data shape
    start = nwb_acq.acquisition['ElectricalSeriesRaw'].starting_time
    step = 1/nwb_acq.acquisition['ElectricalSeriesRaw'].rate
    stop = start + step*nwb_acq.acquisition['ElectricalSeriesRaw'].data.shape[0]
    raw_timestamps = np.arange(start, stop, step)
    
    return raw_timestamps

def timestamps_to_nwb(nwbfile_path, kin_folders, saveData):
    ###### TO NWB ######
    # open the NWB file in r+ mode    

    opened = False
    file_error = True
    while not opened:
        try:
            with NWBHDF5IO(nwbfile_path, 'r+') as io:
                nwbfile = io.read()
                
                file_error = False
                
                nwbfile.keywords = saveData['experiments']
                
                # create a TimeSeries and add it to the processing module 'episode_timestamps_EXPNAME'
                sessPattern = re.compile('[0-9]{3}_acquisition') 
                sessNum = int(re.findall(sessPattern, nwbfile_path)[-1][:3])
                for frame_times, event_info, exp_name, kfold in zip(saveData['frameTimes_byEvent'], saveData['event_info'], saveData['experiments'], kin_folders):

                    timestamps_module_name = 'video_event_timestamps_%s' % exp_name
                    timestamps_module_desc = '''set of timeseries holding timestamps for each behavior/video event for experiment = %s. 
                    Videos are located at %s.
                    This first few elements of the path may need to be changed to the new storage location for the "data" directory.''' % (exp_name, kfold)
    
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