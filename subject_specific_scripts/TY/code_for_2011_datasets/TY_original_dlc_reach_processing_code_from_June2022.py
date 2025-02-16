#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:54:29 2021
@author: Dalton
"""

import pickle
import dill
import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.decomposition import PCA
from scipy.signal import medfilt
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.stats import mode
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


anipose_base = r'C:\Users\Dalton\Documents\lab_files\dlc_temp\anipose_files'

class dpath:
    base = glob.glob(os.path.join(anipose_base, 'trainingsetindex_2'))
    dates = ['2021_02_11']  # for now we can only do one date at a time
    reach_data_storage = r'Z:/marmosets/processed_datasets/reach_and_trajectory_information/%s_reach_and_trajectory_info_updated.pkl' % dates[0].replace('_', '')


class params:    
    fps = 150
    factor_to_cm = 1e-1
    min_cams_threshold = 2
    min_chunk_length = fps * 0.05 #0.04 # 40 ms
    max_gap_to_stitch = fps * 0.2 # 200 ms 
    reproj_threshold = 20
    helmet_percent_tracked_thresh = 0.05
    marker_to_evaluate = 'hand'
    if marker_to_evaluate == 'hand':
        yDir_limits = [-50*factor_to_cm, 125*factor_to_cm]
    elif marker_to_evaluate == 'shoulder':
        yDir_limits  = [-75*factor_to_cm, 125*factor_to_cm]

    blip_thresh = 19
    blip_peak_thresh = 26
    pre_and_post_blip_win = 20 
    retain_thresh = 9.5
    retain_loose_thresh = 14.25
    blip_max_length = fps * 1 #0.5 
    retain_max_length = fps * 1.5 
    pre_and_post_errorPeak_win = 20
    jump_thresh = 9.5
    jump_error_thresh = 40

    speed_thresh = 48


def load_dlc_data(data_dirs):
    data_files = []
    for data_dir in data_dirs:
        data_files.extend(glob.glob(os.path.join(data_dir, '*.csv')))

    dlc = []
    dlc_metadata = []    
    event_info = pd.DataFrame(np.empty((len(data_files), 3)), columns = ['date', 'event', 'recording_marm'])
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

        event_info.iloc[fNum] = [re.findall(namePattern, event_name)[0], 
                                 int(event_name.split('event')[1][:3]), 
                                 False]

    return dlc, dlc_metadata, event_info   

def fix_remaining_marker_jumps(pos_out, pos_fill, pos, marker, meta):

    # remove brief large marker jumps that remain unfiltered
    position_edited = False

    repError = np.array(meta.loc[:, marker+'_error'])                
    repError_interp = pd.Series(repError).interpolate().values
    errorPeakIdxs = find_peaks(repError_interp, distance = 20, height = params.reproj_threshold)[0]
    errorSpike_slices = []
    peaks_to_keep = []
    for pkIdx in errorPeakIdxs:
        if len(errorSpike_slices) == 0 or pkIdx >= errorSpike_slices[-1].stop: 
            append_slice = True
            median_error_thresh = np.nanmedian(meta.loc[:, marker+'_error'])
            prePeak_error  = repError_interp[pkIdx - params.pre_and_post_errorPeak_win : pkIdx]                    
            postPeak_error = repError_interp[pkIdx : pkIdx + params.pre_and_post_errorPeak_win]

            threshMult = 1
            win_extend_pre = 0
            startPoint = np.where(prePeak_error < median_error_thresh * threshMult)[0]
            while len(startPoint) == 0:
                if threshMult <= 3: #median_error_thresh * (threshMult+1) < params.reproj_threshold/4:    
                    threshMult += 1
                else:
                    win_extend_pre += 10
                    threshMult = 1
                    if pkIdx - params.pre_and_post_errorPeak_win - win_extend_pre < 0:
                        append_slice = False
                        break
                    prePeak_error = repError_interp[pkIdx - params.pre_and_post_errorPeak_win - win_extend_pre : pkIdx]                    

                startPoint = np.where(prePeak_error < median_error_thresh * threshMult)[0]

            threshMult = 1
            win_extend_post = 0
            endPoint = np.where(postPeak_error < median_error_thresh * threshMult)[0]
            while len(endPoint) == 0:
                if threshMult <= 3: #median_error_thresh * (threshMult+1) < params.reproj_threshold/4:    
                    threshMult += 1
                else:
                    win_extend_post += 10
                    threshMult = 1
                    if pkIdx + params.pre_and_post_errorPeak_win + win_extend_post > len(repError_interp):
                        append_slice = False
                        break
                    postPeak_error = repError_interp[pkIdx : pkIdx + params.pre_and_post_errorPeak_win + win_extend_post]                    

                endPoint = np.where(postPeak_error < median_error_thresh * threshMult)[0] 
            if append_slice:    
                errorSpike_slices.append(slice(pkIdx - params.pre_and_post_errorPeak_win - win_extend_pre + startPoint[-1], pkIdx + endPoint[0] + 1))
                peaks_to_keep.append(pkIdx)


    # fig = plt.subplots()
    # plt.plot(repError_interp)
    # plt.plot(errorPeakIdxs, repError_interp[errorPeakIdxs], '*')
    # plt.plot(peaks_to_keep, repError_interp[peaks_to_keep], '*')
    # for slc in errorSpike_slices:
    #     print(slc)
    #     plt.plot(np.arange(slc.start, slc.stop), repError_interp[slc], '-r', linewidth=2)

    for jumpSlice in errorSpike_slices:
        pos_jump = pos_out[:, jumpSlice]

        linearInterp = np.empty_like(pos_jump)
        for dim in range(3):
            linearInterp[dim] = np.linspace(pos[dim, jumpSlice.start], pos[dim, jumpSlice.stop-1], pos_jump.shape[-1])

        deviation = np.sum(np.sqrt(np.add(np.square(pos_jump[0] - linearInterp[0]),
                                          np.square(pos_jump[1] - linearInterp[1]),
                                          np.square(pos_jump[2] - linearInterp[2])))) / pos_jump.shape[-1]

        if ~np.isnan(deviation) and (deviation > params.jump_thresh or np.max(repError_interp[jumpSlice]) > params.jump_error_thresh):
            pos_fill[:, jumpSlice] = linearInterp
            position_edited = True

    return pos_fill, position_edited

def prevent_adding_big_jumps_with_interpolation(pos, potential_blip_slices):
    corrected_blip_slices = []
    removed_blip_idxs = []
    for blipIdx, blipSlice in enumerate(potential_blip_slices):
        interpSpeed = pos[:, blipSlice.stop-1] - pos[:, blipSlice.start]
        interpSpeed = np.sqrt(interpSpeed @ interpSpeed)

        if blipIdx == 0:
            nextBlipSlice = potential_blip_slices[blipIdx+1]
            nextInterpSpeed = pos[:, nextBlipSlice.stop-1] - pos[:, nextBlipSlice.start]
            nextInterpSpeed = np.sqrt(nextInterpSpeed @ nextInterpSpeed)
            if interpSpeed > params.speed_thresh and (nextInterpSpeed > params.speed_thresh and nextBlipSlice.start - blipSlice.stop < 30):
                removed_blip_idxs.append(blipIdx)
                continue
        if blipIdx == len(potential_blip_slices) - 1:
            prevBlipSlice = potential_blip_slices[blipIdx-1]
            prevInterpSpeed = pos[:, prevBlipSlice.stop-1] - pos[:, prevBlipSlice.start]
            prevInterpSpeed = np.sqrt(prevInterpSpeed @ prevInterpSpeed)  
            if interpSpeed > params.speed_thresh and (prevInterpSpeed > params.speed_thresh and blipSlice.start - prevBlipSlice.stop < 30):
                removed_blip_idxs.append(blipIdx)
                continue
        else:
            nextBlipSlice = potential_blip_slices[blipIdx+1]
            nextInterpSpeed = pos[:, nextBlipSlice.stop-1] - pos[:, nextBlipSlice.start]
            nextInterpSpeed = np.sqrt(nextInterpSpeed @ nextInterpSpeed)
            prevBlipSlice = potential_blip_slices[blipIdx-1]
            prevInterpSpeed = pos[:, prevBlipSlice.stop-1] - pos[:, prevBlipSlice.start]
            prevInterpSpeed = np.sqrt(prevInterpSpeed @ prevInterpSpeed)

            if interpSpeed > params.speed_thresh and ((   nextInterpSpeed > params.speed_thresh and nextBlipSlice.start - blipSlice.stop < 30) 
                                                      or (prevInterpSpeed > params.speed_thresh and blipSlice.start - prevBlipSlice.stop < 30)):
                removed_blip_idxs.append(blipIdx)
                continue
        corrected_blip_slices.append(blipSlice)

    new_blip_slices = []
    removed_blip_idxs.append(10000)    
    while np.sum(np.diff(removed_blip_idxs) == 1) > 0:
        ct = 0
        for idx, nextIdx in zip(removed_blip_idxs[:-1], removed_blip_idxs[1:]):
            if nextIdx - idx == 1:
                ct += 1 
                continue
            else:
                if ct == 0:
                    del removed_blip_idxs[0]
                else:
                    new_blip_slices.append(slice(potential_blip_slices[removed_blip_idxs[0]].start, 
                                                 potential_blip_slices[removed_blip_idxs[ct]].stop))
                    del removed_blip_idxs[:ct+1]
                break

    corrected_blip_slices = sorted(corrected_blip_slices + new_blip_slices) 

    return corrected_blip_slices

def replace_filtered_blips_with_interpolation(pos_out, pos_fill, pos, marker, meta):

    position_edited = False

    nanIdx = np.where(np.isnan(pos_out[0]))[0]
    if len(nanIdx) == 0:
        return pos_fill, position_edited

    diff_nanIdx = np.where(np.diff(nanIdx) > 1)[0]
    if np.min(nanIdx) > 0:
        diff_nanIdx = np.insert(diff_nanIdx, 0, 0)
    if np.max(nanIdx) < pos_out.shape[-1] - 1:
        diff_nanIdx = np.append(diff_nanIdx, len(nanIdx)-1)
    potential_blip_slices = [] 
    if len(diff_nanIdx) == 1:
        startIdx = nanIdx[0] - 1
        endIdx = nanIdx[-1] + 1
        if endIdx - startIdx < params.blip_max_length: 
            potential_blip_slices.append(slice(startIdx, endIdx))
    else:
        for start, end in zip(diff_nanIdx[:-1], diff_nanIdx[1:]):
            if end - start < params.blip_max_length:
                potential_blip_slices.append(slice(nanIdx[start+1]-1, nanIdx[end]+2))
    if len(potential_blip_slices) >= 2:
        potential_blip_slices = prevent_adding_big_jumps_with_interpolation(pos, potential_blip_slices)

    for blipIdx, blipSlice in enumerate(potential_blip_slices):
        if blipSlice.start < 0 or blipSlice.stop >= pos.shape[-1]:
            continue
        repError_blip = np.array(meta.loc[blipSlice, marker+'_error'])
        pos_blip = pos[:, blipSlice]
        linearInterp = np.empty_like(pos_blip)
        for dim in range(3):
            linearInterp[dim] = np.linspace(pos[dim, blipSlice.start], pos[dim, blipSlice.stop-1], pos_blip.shape[-1])
        deviation = np.sum(np.sqrt(np.add(np.square(pos_blip[0] - linearInterp[0]),
                                          np.square(pos_blip[1] - linearInterp[1]),
                                          np.square(pos_blip[2] - linearInterp[2])))) / pos_blip.shape[-1]
        max_deviation = np.max(np.sqrt(np.add(np.square(pos_blip[0] - linearInterp[0]), np.square(pos_blip[1] - linearInterp[1]),np.square(pos_blip[2] - linearInterp[2]))))

        blip_percent_tracked = np.sum(~np.isnan(repError_blip)) / pos_blip.shape[-1]
        blip_avg_error = np.nansum(repError_blip) / np.sum(~np.isnan(repError_blip))

        if (deviation > params.blip_thresh 
            or (deviation > params.retain_thresh 
                and max_deviation > params.blip_peak_thresh)) and (blip_percent_tracked < 0.5 
                                                                   or blip_avg_error > params.reproj_threshold):
            median_error_thresh = np.nanmedian(meta.loc[:, marker+'_error'])

            if blipSlice.start < params.pre_and_post_blip_win:
                preIdx = 0
            else: 
                preIdx = blipSlice.start - params.pre_and_post_blip_win

            preBlip_error = meta.loc[preIdx : blipSlice.start, marker+'_error']                    
            postBlip_error = meta.loc[blipSlice.stop:blipSlice.stop+20, marker+'_error']

            threshMult = 1
            startPoint = np.where(preBlip_error < median_error_thresh * threshMult)[0]
            while len(startPoint) == 0:
                threshMult += 1
                startPoint = np.where(preBlip_error < median_error_thresh * threshMult)[0]
            new_blip_start = preIdx + startPoint[-1]

            threshMult = 1
            endPoint = np.where(postBlip_error < median_error_thresh * threshMult)[0]
            while len(endPoint) == 0:
                threshMult += 1
                endPoint = np.where(postBlip_error < median_error_thresh * threshMult)[0] 
            new_blip_end = blipSlice.stop + endPoint[0]


            linearInterp = np.empty((3, new_blip_end - new_blip_start))
            for dim in range(3):
                linearInterp[dim] = np.linspace(pos[dim, new_blip_start], 
                                                pos[dim, new_blip_end-1], 
                                                new_blip_end - new_blip_start)

            pos_fill[:, new_blip_start:new_blip_end] = linearInterp

            position_edited = True

    return pos_fill, position_edited


def retain_good_paths_previously_removed(pos_out, pos_fill, pos, marker, meta):
    # retain filtered sections of position that match a linear interpolation closely
    nanIdx = np.where(np.isnan(pos_out[0]))[0]
    if len(nanIdx) == 0:
        return pos_fill

    diff_nanIdx = np.where(np.diff(nanIdx) > 1)[0]
    if np.min(nanIdx) > 0:
        diff_nanIdx = np.insert(diff_nanIdx, 0, 0)
    if np.max(nanIdx) < pos_out.shape[-1] - 1:
        diff_nanIdx = np.append(diff_nanIdx, len(nanIdx)-1)
    potential_blip_slices = [] 
    if len(diff_nanIdx) == 1:
        startIdx = nanIdx[0] - 1
        endIdx = nanIdx[-1] + 1
        if endIdx - startIdx < params.retain_max_length: 
            potential_blip_slices.append(slice(startIdx, endIdx))
    else:
        for start, end in zip(diff_nanIdx[:-1], diff_nanIdx[1:]):
            if end - start < params.retain_max_length:
                potential_blip_slices.append(slice(nanIdx[start+1]-1, nanIdx[end]+2))

    for blipSlice in potential_blip_slices:                        
        pos_blip = pos[:, blipSlice]
        repError_blip = np.array(meta.loc[blipSlice, marker+'_error'])
        linearInterp = np.empty_like(pos_blip)
        for dim in range(3):
            linearInterp[dim] = np.linspace(pos[dim, blipSlice.start], pos[dim, blipSlice.stop-1], pos_blip.shape[-1])
        deviation = np.sum(np.sqrt(np.add(np.square(pos_blip[0] - linearInterp[0]),
                                          np.square(pos_blip[1] - linearInterp[1]),
                                          np.square(pos_blip[2] - linearInterp[2])))) / pos_blip.shape[-1]
        if deviation < params.retain_thresh or (deviation < params.retain_loose_thresh 
                                                and np.nanmean(repError_blip) < params.reproj_threshold 
                                                and np.sum(~np.isnan(repError_blip))/len(repError_blip) > 0.4):
            pos_fill[:, blipSlice] = pos[:, blipSlice]

    return pos_fill

def trim_long_interpolations_at_beginning_and_end(pos, pos_fill):

    # trim ends
    # for adj in [550, 650, 750, 850]:
    #     percent_interp_end = []
    #     for frame in range(pos.shape[-1]-adj):
    #         nInterp = np.sum((pos[0, frame:frame+adj] - pos_fill[0, frame:frame+adj]) == 0)
    #         percent_interp_end.append( nInterp / pos[0, frame:frame+adj].shape[-1])

    #     # trim beginnings
    #     percent_interp_begin = []
    #     for frame in range(pos.shape[-1], 0, -1):
    #         percent_interp_begin.insert(0, np.sum((pos[0, frame-adj:frame] - pos_fill[0, frame-adj:frame]) != 0) / pos[0, frame-adj:frame].shape[-1])

    #     percent_interp_end = savgol_filter(percent_interp_end, 101, 3)   
    #     diff_percent = savgol_filter(np.diff(percent_interp_end*100), 51, 3)

    #     fig = plt.subplots()
    #     # plt.plot(percent_interp_begin)
    #     plt.plot(percent_interp_end)
    #     plt.plot(diff_percent)
    #     plt.plot(1200, percent_interp_end[1200], 'ro')
    #     plt.plot(1200, diff_percent[1200], 'ro')
    #     plt.show()

    percent_interp_end = []
    for frame in range(pos.shape[-1]):
        nInterp = np.sum((pos[0, frame:] - pos_fill[0, frame:]) == 0)
        percent_interp_end.append( nInterp / pos[0, frame:].shape[-1])

    percent_interp_end = savgol_filter(percent_interp_end, 101, 3)    
    # diff_percent = savgol_filter(np.diff(percent_interp_end*100), 51, 3)

    try:
        cutIdx = np.where(percent_interp_end < 0.25 * np.max(percent_interp_end[100:-100]))[0][0]

        pos_fill[:, cutIdx:] = np.nan

        # fig = plt.subplots()
        # # plt.plot(percent_interp_begin)
        # plt.plot(percent_interp_end)
        # # plt.plot(diff_percent)
        # plt.plot(cutIdx, percent_interp_end[cutIdx], 'ro')
        # # plt.plot(1200, diff_percent[1200], 'ro')
        # plt.show()
        # # minima = find_peaks(percent_interp_end)
    except:
        pass



    return pos_fill

def filter_dlc(dlc, dlc_metadata, event_info, events = None):
    dlc_filtered = []
    for eventNum, (pos, meta) in enumerate(zip(dlc, dlc_metadata)):

        if events is not None and eventNum+1 not in events: 
            continue

        marker_names = meta.columns[slice(0, len(meta.columns), 3)]
        marker_names = [name[:-6] for name in marker_names]
        pos_first_filter = pos.copy()
        pos_out      = np.full_like(pos, np.nan)

        # iterate over markers to filter out bad portions of trajctories
        for mNum, marker in enumerate(marker_names):

            # find frames where the reprojection error is worse than a specified threshold or 
            # the number of cams is below a threshold (min_cams_threshold=2 unless good reason to change it)
            # and set position in those frames to np.nan
            # fig, ax = plt.subplots()
            # ax.plot(meta.loc[:, marker+'_error'])
            # ax.plot(range(meta.shape[0]), np.repeat(params.reproj_threshold, meta.shape[0]), '-r')
            # plt.show()

            filterOut_idx = np.union1d(np.where(meta.loc[:, marker+'_error'] > params.reproj_threshold)[0],
                                       np.where(meta.loc[:, marker+'_ncams'] < params.min_cams_threshold)[0])
            pos_first_filter[mNum, :, filterOut_idx] = np.nan

            if np.sum(~np.isnan(pos_first_filter[mNum])) == 0:
                continue

            # Find beginning and end of each 'chunk' that remains after filtering
            gap_idxs = np.where(np.isnan(pos_first_filter[mNum, 0]))[0]
            if len(gap_idxs) == 0:
                chunk_starts = np.array([0])
                chunk_ends = np.array([pos_first_filter.shape[-1] - 1])
            else:
                chunk_starts = gap_idxs[np.hstack((np.diff(gap_idxs) > 1, False))] + 1
                chunk_ends   = gap_idxs[np.hstack((False, np.diff(gap_idxs) > 1))] - 1 
                if gap_idxs[0] != 0:
                    chunk_starts = np.hstack((0, chunk_starts))
                    chunk_ends   = np.hstack((gap_idxs[0] - 1, chunk_ends))
                if gap_idxs[-1] != pos_first_filter.shape[-1] - 1:
                    chunk_starts = np.hstack((chunk_starts, gap_idxs[-1] + 1))
                    chunk_ends   = np.hstack((chunk_ends  , pos_first_filter.shape[-1] - 1))

            left_gap_lengths = np.hstack((chunk_starts[0], chunk_starts[1:] - chunk_ends[:-1] - 1))

            # Write chunk and gap info into readable format. Note that to grab a particular chunk you would index 
            # with [chunk_info.start : chunk_info.end + 1]
            chunk_info = pd.DataFrame(data = zip(chunk_starts, 
                                                 chunk_ends, 
                                                 chunk_ends - chunk_starts + 1, 
                                                 left_gap_lengths), 
                                      columns=['start', 
                                               'end', 
                                               'chunk_length', 
                                               'prev_gap_length'])

            # remove chunks shorter than the minimum chunk_length
            while np.sum(chunk_info.chunk_length < params.min_chunk_length) > 0:
                idx = chunk_info.index[chunk_info.chunk_length < params.min_chunk_length][0]
                if idx < chunk_info.index.max():
                    chunk_info.loc[idx + 1, 'prev_gap_length'] += chunk_info.loc[idx, 'prev_gap_length'] + chunk_info.loc[idx, 'chunk_length']
                    chunk_info = chunk_info.drop(idx, axis = 0)     
                    chunk_info = chunk_info.reset_index(drop=True)
                else: 
                    chunk_info = chunk_info.drop(idx, axis = 0)

            # stitch together chunks with gaps shorter than the max_gap_to_stitch parameter
            while np.sum(chunk_info.prev_gap_length[1:] < params.max_gap_to_stitch) > 0:
                idx  = chunk_info.index[chunk_info.prev_gap_length < params.max_gap_to_stitch]
                idx_after_first_chunk = np.nonzero(idx)[0]
                if len(idx_after_first_chunk) != 0:
                    idx = idx[idx_after_first_chunk[0]]

                    chunk_info.loc[idx-1, 'end']          = chunk_info.loc[idx, 'end']
                    chunk_info.loc[idx-1, 'chunk_length'] = chunk_info.loc[idx-1, 'end'] - chunk_info.loc[idx-1, 'start'] + 1 
                    chunk_info = chunk_info.drop(idx, axis=0)
                    chunk_info = chunk_info.reset_index(drop=True)

            # produce a filtered position with only the remaining stitched-together chunks
            filtered_chunk_idxs = []
            for index, chunk in chunk_info.iterrows():
                filtered_chunk_idxs.extend(list(range(chunk.start, chunk.end+1)))
            pos_out[mNum, :, filtered_chunk_idxs] = pos[mNum, :, filtered_chunk_idxs]

            # NOW work on specific filtering issues that are remaining!
            if marker not in ['origin', 'x', 'y']:

                pos_fill = np.copy(pos_out[mNum])

                print((eventNum+1, marker, 'jumps'))
                pos_fill, jumps_edited = fix_remaining_marker_jumps(pos_out[mNum], pos_fill, pos[mNum], marker, meta)

                print((eventNum+1, marker, 'blips'))
                pos_fill, blips_edited = replace_filtered_blips_with_interpolation(pos_out[mNum], pos_fill, pos[mNum], marker, meta)                

                print((eventNum+1, marker, 'retain'))
                pos_fill = retain_good_paths_previously_removed(pos_out[mNum], pos_fill, pos[mNum], marker, meta)

                print((eventNum+1, marker, 'trim'))
                pos_fill = trim_long_interpolations_at_beginning_and_end(pos[mNum], pos_fill)

                for dim in range(3):
                    try:
                        smoothIdx = np.where(~np.isnan(pos_fill[dim]))[0]
                        diff_smoothIdx = np.where(np.diff(smoothIdx) > 1)[0]

                        smoothChunks = []
                        if len(smoothIdx) == len(pos_fill[dim]):
                            smoothChunks.append(slice(0, len(pos_fill[dim])))
                        elif len(diff_smoothIdx) == 0 and len(smoothIdx) < len(pos_fill[dim]):
                            smoothChunks.append(slice(smoothIdx[0], smoothIdx[-1]+1))
                        elif len(diff_smoothIdx) > 0:
                            for start, end in zip(diff_smoothIdx[:-1], diff_smoothIdx[1:]):
                                if end - start > 11:
                                    smoothChunks.append(slice(smoothIdx[start+1], smoothIdx[end]+1))
                        for smoothChunk in smoothChunks:
                            pos_fill[dim, smoothChunk] = savgol_filter(pos_fill[dim, smoothChunk], 11, 3)
                    except:
                        pass

                pos_out[mNum] = pos_fill         
                # if marker == params.marker_to_evaluate:
                #     fig, axs = plt.subplots(2, 1)
                #     # axs[0].plot(np.arange(1, pos_out.shape[-1] + 1), pos_out[mNum, 1], linewidth=3)
                #     axs[0].plot(np.arange(1, pos.shape[-1] + 1), pos[mNum, 1], linewidth=2)
                #     axs[0].plot(np.arange(1, pos.shape[-1] + 1), pos_fill[1], linewidth=2)
                #     axs[0].plot(np.arange(1, pos.shape[-1] + 1), np.repeat(0, pos.shape[-1]), '-r')
                #     axs[0].set_title('Event %d - %s' % (eventNum+1, marker))
                #     axs[0].set_ylim(params.yDir_limits)

                #     axs[1].plot(meta.loc[:, marker+'_error'])
                #     axs[1].plot(range(meta.shape[0]), np.repeat(params.reproj_threshold, meta.shape[0]), '-r')

                #     plt.show()

        head_marker_idxs = [idx for idx, name in enumerate(marker_names) if 'head' in name]
        head_pos = pos_out[head_marker_idxs]
        head_tracked_percent = np.sum(np.all(~np.isnan(head_pos[:, 0]), axis = 0)) / head_pos.shape[-1]

        if head_tracked_percent > params.helmet_percent_tracked_thresh:
            event_info.loc[eventNum, 'recording_marm'] = True    

        dlc_filtered.append(pos_out * params.factor_to_cm)

    markerIdx = [idx for idx, name in enumerate(marker_names) if name == params.marker_to_evaluate][0]

    return dlc_filtered, markerIdx

def evaluate_labeling_quality(dlc_filtered, dlc, dlc_metadata, event_info, plotSet = None, plotEvents = None):
    if plotSet is None and plotEvents is None:
        plotSet = [0, np.sum(event_info.recording_marm)]

    if plotSet is not None and plotEvents is None:
        ct = 0
        for eventNum, (pos_filt, pos, meta, recMarm) in enumerate(zip(dlc_filtered, dlc, dlc_metadata, event_info.recording_marm)):
            if recMarm: 
                if ct >= plotSet[0] and ct < plotSet[1]:
                    marker_names = meta.columns[slice(0, len(meta.columns), 3)]
                    marker_names = [name[:-6] for name in marker_names]

                    markerIdx = [idx for idx, name in enumerate(marker_names) if name == params.marker_to_evaluate][0]
                    marker = marker_names[markerIdx]

                    fig, axs = plt.subplots(2, 1)
                    axs[0].plot(np.arange(1, pos.shape[-1] + 1), pos[markerIdx, 1])
                    axs[0].plot(np.arange(1, pos_filt.shape[-1] + 1), pos_filt[markerIdx, 1])
                    axs[0].plot(np.arange(1, pos.shape[-1] + 1), np.repeat(0, pos.shape[-1]))
                    axs[0].set_title('Event %d - %s' % (eventNum+1, marker))
                    axs[0].set_ylim(params.yDir_limits)

                    if params.marker_to_evaluate != 'hand':
                        markerIdx = [idx for idx, name in enumerate(marker_names) if name == 'hand'][0]  
                        axs[0].plot(np.arange(1, pos_filt.shape[-1] + 1), pos_filt[markerIdx, 1])

                    axs[1].plot(meta.loc[:, marker+'_error'])
                    axs[1].plot(range(meta.shape[0]), np.repeat(params.reproj_threshold, meta.shape[0]), '-r')

                    plt.show()

                ct += 1
    elif plotEvents is not None and plotSet is None:
        for eventNum, (pos_filt, pos, meta, recMarm) in enumerate(zip(dlc_filtered, dlc, dlc_metadata, event_info.recording_marm)):
            if recMarm and eventNum+1 in plotEvents: 
                marker_names = meta.columns[slice(0, len(meta.columns), 3)]
                marker_names = [name[:-6] for name in marker_names]

                markerIdx = [idx for idx, name in enumerate(marker_names) if name == params.marker_to_evaluate][0]
                marker = marker_names[markerIdx]

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(np.arange(1, pos.shape[-1] + 1), pos[markerIdx, 1], '-b')
                axs[0].plot(np.arange(1, pos_filt.shape[-1] + 1), pos_filt[markerIdx, 1], '-', color='orange')
                axs[0].plot(np.arange(1, pos.shape[-1] + 1), np.repeat(0, pos.shape[-1]))
                axs[0].set_title('Event %d - %s' % (eventNum+1, marker))
                axs[0].set_ylim(params.yDir_limits)
                if params.marker_to_evaluate != 'hand':
                    markerIdx = [idx for idx, name in enumerate(marker_names) if name == 'hand'][0]  
                    axs[0].plot(np.arange(1, pos_filt.shape[-1] + 1), pos_filt[markerIdx, 1])
                axs[1].plot(meta.loc[:, marker+'_error'])
                axs[1].plot(range(meta.shape[0]), np.repeat(params.reproj_threshold, meta.shape[0]), '-r')

                plt.show()

def compute_derivatives(marker_pos, smooth = True):
    marker_vel = np.diff(marker_pos, axis = -1) * params.fps
    if smooth:
        for dim in range(3):
            marker_vel[dim] = gaussian_filter(marker_vel[dim], sigma=1.5)

    marker_acc = np.diff(marker_vel, axis = -1) * params.fps
    if smooth:
        for dim in range(3):
            marker_acc[dim] = gaussian_filter(marker_acc[dim], sigma=1.5)

    return marker_vel, marker_acc

def delete_overlapping(reach_timing):
    # delete overlapping reaching epochs from reach timing. adapted from ephyviewer epoch.
    ep_starts = reach_timing['starts']
    ep_stops = reach_timing['stops']
    eventNum = reach_timing['event']
    ep_durations = []

    ep_timing = zip(ep_stops, ep_starts)
    for list1_i, list2_i in ep_timing:
        ep_durations.append(list1_i-list2_i)

    peaks = reach_timing['peaks']
    inds = range(len(ep_durations))

    # adapted from ephyviewer epoch
    for i in range(len(inds)-1):

        # if two sequentially adjacent epochs with the same label
        # overlap or have less than 1 microsecond separation, merge them
        if ep_starts[inds[i+1]] - ep_stops[inds[i]] < 1e-6:

            # stretch the second epoch to cover the range of both epochs
            ep_starts[inds[i+1]] = min(ep_starts[inds[i]], ep_starts[inds[i+1]])
            ep_stops[inds[i+1]] = max(ep_stops[inds[i]], ep_stops[inds[i+1]])
            ep_durations[inds[i+1]] = ep_stops[inds[i+1]] - ep_starts[inds[i+1]]

            # flag epochs for deleting
            ep_durations[inds[i]] = -1 # non-positive duration flags this epoch for clean up

    to_keep_bool = np.array(ep_durations) >= 0
#     set_trace()
    reach_timing_cleaned = {'event': eventNum,
                            'starts':np.array(ep_starts)[to_keep_bool].tolist(),
                            'stops': np.array(ep_stops)[to_keep_bool].tolist(),
                            'durations': np.array(ep_durations)[to_keep_bool].tolist(),
                            'peaks': peaks}

    return reach_timing_cleaned

def get_reach_timing_from_event(hand_pos, eventNum, method = 2, include_reaches_missing_start_or_stop = False, return_all = False):

    #     '''
    #          Given 3D hand data, for each reach, ret urns start and end frame/time
    #          Example input it should be able to handle: 
    #                  data[event_name]['df_3D'] = df_3D_vel_added
    #                  hand_data = data_mth_marm[event_name_]['df_3D'][scorer_cam1][marm_bp]
    #          return: reach_timing dict containing an len num_reaches array each for starts and stops 
    #     '''

    hand_vel, hand_acc = compute_derivatives(hand_pos)

    event_time = hand_pos.shape[-1] / params.fps
    hand_speed = np.sqrt(np.square(hand_vel).sum(axis = 0))
    # event_time = hand_data.index.values/params.fps
    # hand_speed = np.sqrt(np.square(hand_data[['vx', 'vy', 'vz']]).sum(axis=1)).values * params.fps

    if method == 1: # method one: y position thresholds
        print('identifying reach starts and stops based on y position threshold')
        # define position threshold 
        position_threshold = 0
        # filter on hand position in y
        passed = hand_data['y'] > position_threshold
        passed_i16 = passed.astype(np.int16)
        passed_diff = np.diff(passed_i16)
        reach_starts = np.where(passed_diff == 1)
        reach_starts = reach_starts[0]
        reach_stops = np.where(passed_diff == -1)
        reach_stops = reach_stops[0]
        reach_timing = {'starts':reach_starts, 'stops':reach_stops}

    elif method == 2: # method two: peak finding and noise thresholds in velocity features
        print('identifying reach starts and stops based peak finding in y and z velocity')
        # find reach start/movement onset
        yz_noise_thresh = 5
        peak_min = 15
        yz_vel = hand_data['vy']+hand_data['vz']*params.fps
        peaks = signal.find_peaks(yz_vel, height = peak_min)
        peaks = peaks[0]
        yz_thresh_crossing = yz_vel > yz_noise_thresh
        yz_thresh_crossing = yz_thresh_crossing.astype(np.int16)
        n2p = np.where(np.diff(yz_thresh_crossing) == 1)
        n2p = n2p[0]
        p2n = np.where(np.diff(yz_thresh_crossing) == -1)
        p2n = p2n[0]        
        reach_starts = [0]*len(peaks)
        reach_stops = [0]*len(peaks)
        for peak_num, peak_frame in enumerate(peaks):
            thresh_crossing_before_peak = np.max(n2p[n2p <= peak_frame])
            reach_starts[peak_num] = thresh_crossing_before_peak
            thresh_crossing_after_peak = np.min(p2n[p2n >= peak_frame])
            reach_stops[peak_num] = thresh_crossing_after_peak
        reach_timing = {'starts':reach_starts, 'stops':reach_stops}

    elif method == 3: # find hand speed threshold crossings before and after y position peaks
        print('method 3: reach parsing based on hand speed threshold crossings around y position peaks')
        y_position_threshold = 1.5 # cm
        y_peak_threshold = 3.85 # cm
        speed_threshold = 5 # cm/sec
        # id peaks in y position
        peaks = find_peaks(hand_pos[1], 
                           height = y_peak_threshold, 
                           distance = 75, 
                           prominence=1, 
                           wlen = 401)

        peaks = peaks[0].tolist()
        # prominences = peak_prominences(hand_pos[1], peaks, wlen = 201)
        # widths      = peak_widths(hand_pos[1], peaks, prominence_data = prominences, wlen = 101)
        # print((peaks, prominences[0]))
        # id threshold crossings of y position as candidate reach starts and stops
        passed = hand_pos[1] > y_position_threshold
        passed_i16 = passed.astype(np.int16)
        passed_diff = np.diff(passed_i16)
        reach_start_candidates = np.where(passed_diff == 1)[0]
        reach_stop_candidates = np.where(passed_diff == -1)[0]
        # refine starts and stops with hand speed threshold crossings
        speed_thresh_crossing = hand_speed > speed_threshold
        speed_thresh_crossing = speed_thresh_crossing.astype(np.int16)
        n2p = np.where(np.diff(speed_thresh_crossing) == 1)[0] # negative to positive crossings
        p2n = np.where(np.diff(speed_thresh_crossing) == -1)[0] # positive to negative crossings

        # checks if reaches are present
        if len(reach_start_candidates) != 0 or len(reach_stop_candidates) != 0:
            reaches_present = True
            print('reaches may be present in event')
            if len(peaks) == 0:
                print('... but, no prominent peaks in y-position. so maybe not')
        else:
            reaches_present = False
            print('reaches not present in event')

        if reaches_present == True:    
            truncated_start = False
            truncated_end = False
            # check data for edge cases

            if len(reach_stop_candidates) > 0 and len(reach_start_candidates) > 0: 
                if reach_stop_candidates[0]<= reach_start_candidates[0] or n2p[0] >= reach_start_candidates[0]:
                    print(f'--> first reach starts before event <--')
                    truncated_start = True
                    if reach_stop_candidates[0] < reach_start_candidates[0] \
                    and len(reach_start_candidates) > 1 and len(reach_stop_candidates) >1:
                        reach_start_candidates = np.concatenate(([-1], reach_start_candidates.tolist())).tolist()
                if reach_start_candidates[-1] >= reach_stop_candidates[-1]:
                    print(f'-->last reach ends after event <--')
                    truncated_end = True
                    reach_stop_candidates = np.concatenate((reach_stop_candidates.tolist(), [hand_pos.shape[-1]])).tolist()
            else:
                print(f'--> there is no reach start or reach stop candidate <--')  

            if include_reaches_missing_start_or_stop and len(reach_stop_candidates) == 0 and len(peaks) > 0:
                truncated_end = True
                reach_stop_candidates = [hand_pos.shape[-1]]    
            if include_reaches_missing_start_or_stop and len(reach_start_candidates) == 0 and len(peaks) > 0:
                truncated_start = True
                reach_start_candidates = [0]  

            # TODO - check for, and handle, nans

            medfiltered_pos = medfilt(hand_pos[1], 31)
            max_err  = np.nanmax (abs(hand_pos[1] - medfiltered_pos))
            mean_err = np.nanmean(abs(hand_pos[1] - medfiltered_pos))

            # id reach starts
            print(('start', reach_start_candidates))
            reach_starts = [0]*len(reach_start_candidates)
            for reach_num, candidate_start_frame in enumerate(reach_start_candidates):
                if truncated_start == True and reach_num == 0: # set start to -1 to flag before event start
#                   prior_min = np.min(hand_data['y'].iloc[0:candidate_start_frame])
                    if include_reaches_missing_start_or_stop:
                        reach_starts[reach_num] = 0   
                    print(f'setting reach {reach_num} start to -1 for before event_start')
                elif candidate_start_frame >= n2p[reach_num]: # simple case
                    print(f'adjusting reach {reach_num} start to prior speed threshold crossing')
                    thresh_crossing_before_peak = np.max(n2p[n2p <= candidate_start_frame])
                    reach_starts[reach_num] = thresh_crossing_before_peak
                elif include_reaches_missing_start_or_stop and mean_err < .1:
                    reach_starts[reach_num] = candidate_start_frame
                else:
                    print('un-considered start condition')

            print(('stop', reach_stop_candidates))
            # id reach stops
            reach_stops = [0]*len(reach_stop_candidates)
            for reach_num, candidate_stop_frame in enumerate(reach_stop_candidates):
                # if truncated_start == True and reach_num == 0:
                #     print(f'adjusting reach {reach_num} stop frame for truncated stop, TODO')
                if truncated_end == True and reach_num == len(reach_starts)-1:
                    if include_reaches_missing_start_or_stop and mean_err < .1:
                        reach_stops[reach_num] = candidate_stop_frame
                    print(f'reach {reach_num} may end after event, set stop to last frame')
                elif candidate_stop_frame <= p2n[-1]:  #else:
                    print(f'adjusting reach {reach_num} stop to next speed threshold crossing')
                    try:
                        thresh_crossing_after_peak = np.min(p2n[p2n >= candidate_stop_frame])
                        reach_stops[reach_num] = thresh_crossing_after_peak
                    except ValueError:
                        reach_stops[reach_num] = np.nan
                elif include_reaches_missing_start_or_stop and mean_err < .1:
                    reach_stops[reach_num] = candidate_stop_frame

            if len(reach_stops) > len(reach_starts):
                reach_stops = reach_stops[len(reach_stops) - len(reach_starts) : ]
            reaches_to_remove = []
            for reach_num, (start, stop) in enumerate(zip(reach_starts, reach_stops)):
                if np.isnan(start) or np.isnan(stop):
                    reaches_to_remove.append(reach_num)
                    continue
                if not any(peak in range(start, stop) for peak in peaks): # check if any peaks were detected within the potential reach
                    reaches_to_remove.append(reach_num)

            reach_starts = [start for reach_num, start in enumerate(reach_starts) if reach_num not in reaches_to_remove]
            reach_stops  = [stop  for reach_num, stop  in enumerate(reach_stops)  if reach_num not in reaches_to_remove]

            thresholds = {'y_position':y_position_threshold, 'y_peak':y_peak_threshold, 'speed':speed_threshold}
            reach_timing = {'event': eventNum, 'starts':reach_starts, 'stops':reach_stops, 'peaks': peaks} 

        elif reaches_present == False:
            thresholds = {'y_position':y_position_threshold, 'y_peak':y_peak_threshold, 'speed':speed_threshold}
            reach_timing = {'event': eventNum, 'starts':[], 'stops':[], 'peaks': []}

    elif method == 4: # wavelets
        print('method 4: based on CWT not implemented')

    # delete overlapping reaching epochs
    reach_timing = delete_overlapping(reach_timing)

    if return_all == False:
        return reach_timing
    elif return_all == True:
        if method == 1: 
            return reach_timing
        elif method == 2:
            return reach_timing, yz_vel, yz_thresh_crossing
        elif method == 3:
            return reach_timing, hand_speed, thresholds
        elif method == 4:
            return 

def plot_reach_timing(event_name, hand_pos, reach_timing, hand_speed, thresholds):
    # create figure to show data relevant to reach parsing (start, stop, peaks)
    fig, ax1 = plt.subplots(figsize = (14,3))

    # plot hand speed
    color = 'tab:red'
    ax1.set_ylabel('hand speed (cm/s)', color=color)
    ax1.plot(hand_speed, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
#     set_trace()
    ax1.plot(np.ones(len(hand_speed))*thresholds['speed'], \
             linestyle='dashed', linewidth=1.5, color='tab:red', label = 'speed thresh')

    # plot hand position in y
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('y position (cm)', color=color)  # we already handled the x-label with ax1
    ax2.plot(hand_pos[1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(np.ones(len(hand_pos[1]))*thresholds['y_position'], \
             linestyle='dotted', linewidth=1.5, color='tab:blue', label = 'y position thresh')
    ax2.plot(np.ones(len(hand_pos[1]))*thresholds['y_peak'], \
             linestyle='dashed', linewidth=1.5, color='tab:blue', label = 'y peak thresh')

    # plot reaching epochs
    if reach_timing != 0:   # check if reach timing was provided
        reach_starts = reach_timing['starts']
        reach_stops = reach_timing['stops']
        if len(reach_starts) == len(reach_stops):
            for reach in range(len(reach_starts)):
                ax1.axvspan(reach_starts[reach], reach_stops[reach], color = 'k', alpha = 0.05)
        else:
            print('unequal number of reach starts and stops')
        ax2.scatter(reach_timing['peaks'], hand_pos[1, reach_timing['peaks']], \
                     marker = '*', s = 200,color = 'red', label = 'y position peak')

    ax1.legend(loc='upper right', frameon = False)
    ax2.legend(loc='upper left', frameon = False)
    ax1.set_title(event_name);
    fig.tight_layout()
    plt.show()

    return fig

def plot_event_3Dhand(event_data_3Dhand, event_name, window = [], reach_timing = []):

    fps = 150
    df = event_data_3Dhand
    event_time = event_data_3Dhand.index.values/fps

    if len(window) == 2:
        win_start  = np.where(event_time == window[0])[0][0]
        win_end =np.where(event_time == window[1])[0][0]
    elif bool(window) == False:
        win_start = 0
        win_end = len(event_time)-1

    if bool(reach_timing) == True:
        reach_starts = reach_timing['starts']
        reach_stops = reach_timing['stops']
        peaks = reach_timing['peaks']
    else:
        reach_starts = []
        reach_stops = []
        peaks = []


    speed = np.sqrt(np.square(event_data_3Dhand[['vx', 'vy', 'vz']]).sum(axis=1)).values

    # plot features and x,y,z position v time
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, frameon = False, figsize = (16,6))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)


    ax1.plot(event_time[win_start:win_end], df['x'][win_start:win_end]- df['x'][win_start:win_end].mean(axis = 0),\
             color = 'r', label = 'x')
    ax2.plot(event_time[win_start:win_end], df['vx'][win_start:win_end]*150, color = 'crimson', label = 'vx')
    ax1.plot(event_time[win_start:win_end], df['y'][win_start:win_end]- df['y'][win_start:win_end].mean(axis = 0),\
             color = 'g', label = 'y')
    ax2.plot(event_time[win_start:win_end], df['vy'][win_start:win_end]*150, color = 'darkgreen', label = 'vy')
    ax1.plot(event_time[win_start:win_end], df['z'][win_start:win_end]- df['z'][win_start:win_end].mean(axis = 0),\
             color = 'b', label = 'z')
    ax2.plot(event_time[win_start:win_end], df['vz'][win_start:win_end]*150, color = 'darkblue', label = 'vz')
    ax3.plot(event_time[win_start:win_end],speed[win_start:win_end]*150, label = 'speed')

    if len(reach_starts) != 0:   # check if reach timing was provided
        if len(reach_starts) == len(reach_stops):
            for reach in range(len(reach_starts)):
                ax1.axvspan(reach_starts[reach]/fps, reach_stops[reach]/fps, color = 'k', alpha = 0.05)
                ax2.axvspan(reach_starts[reach]/fps, reach_stops[reach]/fps, color = 'k', alpha = 0.05)
                ax3.axvspan(reach_starts[reach]/fps, reach_stops[reach]/fps, color = 'k', alpha = 0.05)

    if len(peaks) != 0: # check if peaks in y position have been annotated
        y_position = df['y'][win_start:win_end]
        ax1.scatter(np.array(peaks)/fps, y_position.iloc[np.array(peaks)],marker = '*', s = 200,color = 'red' )  

    ax1.set_ylim([-10,15])
    ax1.set_xlim([win_start/fps, win_end/fps])
    ax1.set_title(event_name);
    ax1.legend(loc='upper right',frameon=False, fontsize=14)
    ax1.axes.xaxis.set_visible(False)
    ax2.legend(loc='upper right',frameon=False, fontsize=14)
    ax2.set_xlim([win_start/fps, win_end/fps])
    ax2.axes.xaxis.set_visible(False)
    ax3.legend(loc='upper right',frameon=False, fontsize=14)
    ax3.set_xlim([win_start/fps, win_end/fps])
    ax3.set_xlabel('time (sec)', fontsize=14);
    plt.show()

    return fig


def get_3d_reach_data(dlc_filtered, markerIdx, include_reaches_missing_start_or_stop = False, events = None, plot = False):

    if events is None:
        events_to_check = dlc_filtered
        eventNums = range(1, len(dlc_filtered) + 1)
    else:
        events_to_check = [pos for eventNum, pos in enumerate(dlc_filtered) if eventNum+1 in events]
        eventNums = events

    all_reach_timing = []
    for eventNum, pos in zip(eventNums, events_to_check):

        print('\n')
        print(['data for event %d is good, getting reach timing' % eventNum])        
        reach_timing, hand_speed, thresholds = get_reach_timing_from_event(pos[markerIdx],
                                                                           eventNum,
                                                                           method = 3, 
                                                                           include_reaches_missing_start_or_stop = include_reaches_missing_start_or_stop, 
                                                                           return_all = True)
        # report reach timing
        print(reach_timing)

        if plot:
            reach_timing_fig = plot_reach_timing('Event %d' % eventNum, pos[markerIdx], 
                                   reach_timing, hand_speed, thresholds)
            # plot event data for inspection
            # fig = plot_event_3Dhand(pos[markerIdx], 'Event %d' % eventNum, window = [], reach_timing = reach_timing)

        all_reach_timing.append(reach_timing)

    return all_reach_timing

def add_trajectories_to_reach_data(dlc_filtered, reach_data, dlc_metadata):

    for event_data in reach_data: 
        eventNum = event_data['event']

        pos = dlc_filtered[eventNum - 1]
        vel = np.empty_like(pos)[..., :-1]
        for mIdx, markerPos in enumerate(pos):
            vel[mIdx], tmp_acc = compute_derivatives(markerPos, smooth=True) 

        event_data['position'] = pos
        event_data['velocity'] = vel

        meta = dlc_metadata[eventNum - 1]
        marker_names = meta.columns[slice(0, len(meta.columns), 3)]
        event_data['marker_names'] = [name[:-6] for name in marker_names]

    return reach_data

if __name__ == "__main__":

    for base in dpath.base:
        print('\n\n\n' + base + '\n\n\n')

        data_dirs = []
        for date in dpath.dates:
            data_dirs.append(os.path.join(base, date, 'pose-3d'))

        dlc, dlc_metadata, event_info = load_dlc_data(data_dirs)
        dlc_filtered, markerIdx = filter_dlc(dlc, dlc_metadata, event_info, events = None)

        dlc = [pos*params.factor_to_cm for pos in dlc]
        evaluate_labeling_quality(dlc_filtered, dlc, dlc_metadata, event_info, 
                                  plotSet = None, 
                                  plotEvents = [22, 37, 85, 88, 146, 160, 166, 171] )      
        # evaluate_labeling_quality(dlc_filtered, dlc, dlc_metadata, event_info, 
        #                           plotSet = None, 
        #                           plotEvents = [4, 5, 7, 8, 10, 13, 16, 19, 22, 24, 26, 32, 33, 37, 39, 42, 47, 49, 53, 55, 56, 58, 60, 68, 71, 73, 74, 76])
        #                         [4, 5, 7, 8, 10, 12, 13, 16, 19, 22, 24, 26, 32, 33, 37, 39, 42, 47, 49, 53, 55, 56, 58, 60, 68, 71, 73, 74, 76]                          
        #                         [78, 79, 80, 82, 83, 85, 86, 88, 89, 92, 96,100, 102, 110, 111, 112, 113, 118, 121, 124, 125, 126, 127, 128]
        #                         [131, 134, 138, 139, 140, 142, 145, 146, 148, 151, 154, 155, 156, 157, 161, 163, 165, 167, 172, 173, 175, 179, 180, 181, 184, 186, 189, 191]

        #%%    
        reach_data = get_3d_reach_data(dlc_filtered, markerIdx, include_reaches_missing_start_or_stop = True, 
                                          events = [4  , 5  , 7  , 8  , 10, 13 , 16 , 19 , 22, 26 , 32 , 33 , 39 , 42 , 47 , 
                                                    49 , 53 , 55 , 56, 58 , 60, 68 , 71 , 73 , 74, 76 , 78 , 79 , 80 , 83 , 86 , 
                                                    89 , 92 , 96 , 100, 102, 110, 111, 112, 113, 116, 118, 121, 124, 125, 126, 
                                                    127, 128, 131, 133, 134, 138, 139, 140, 142, 145, 148, 151, 154, 155, 156, 157, 160, 161, 163, 165, 
                                                    166, 167, 171, 172, 173, 175, 180, 181, 184, 186, 189, 190, 191], 
                                          plot = False)  
        # reach_data = get_3d_reach_data(dlc_filtered, markerIdx, include_reaches_missing_start_or_stop = True, 
        #                                  events = [22, 60, 74, 116, 133, 138, 156, 157, 160, 163, 166, 171, 172, 181, 184, 186, 189, 190, 191], 
        #                                  plot = True) 

        reach_data = add_trajectories_to_reach_data(dlc_filtered, reach_data, dlc_metadata)





        # [4  , 5  , 7  , 8  , 10 , 13 , 16 , 19 , 26 , 32 , 33 , 39 , 42 , 47 , 49 , 53 , 55 , 56 ]
        # [58 , 68 , 71 , 73 , 76 , 78 , 79 , 80 , 83 , 86 , 89 , 92 , 96 , 100, 102, 110, 111, 112] 
        # [113, 118, 121, 124, 125, 126, 127, 128, 131, 134, 139, 140, 142, 145, 148, 151, 154, 155]
        # [161, 165, 167, 172, 173, 175, 180, 184, 186, 189, 191]
        # 
        # events with no reach_start = [5, 13, 32, 111, 121, 125, 128, 161, 167, 172, 189] - SOLVED
        # bad events [24, 37]
        # events to look at later [89, 110, 113, 134, 142] - has nans, but pass tests and have identified reaches!

        with open(dpath.reach_data_storage, 'wb') as fp:
            dill.dump(reach_data, fp, recurse=True)