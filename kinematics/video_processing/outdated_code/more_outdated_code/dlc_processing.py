#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:42:28 2020

@author: daltonm
"""

###### Need to get trajectories from DLC_3d project on marmsComp

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import copy
import pickle
import os
from scipy.io import savemat
import subprocess

operSystem = 'windows' # can be windows or linux

processedData_dir     = '/marmosets/processed_datasets/2019_11_26/'
traj_dir              = '/marmosets/deeplabcut_results/PT_foraging-Dalton-2019-12-03/all_trajectories/'
traj_storage_filename = '2019_11_26_foraging_trajectories_session_1_2_3_shuffle1_330000'

class params: 
    if operSystem == 'windows':
        traj_path          = os.path.join(r'Z:', traj_dir)
        traj_processedPath = os.path.join(r'Z:', processedData_dir, traj_storage_filename)
    elif operSystem == 'linux':
        traj_path          = os.path.join('/media/CRI', traj_dir)   
        traj_processedPath = os.path.join('/media/CRI', processedData_dir)
        tmpStorage         = os.path.join('/home/marmosets/Documents/tmpProcessStorage/', traj_storage_filename) 
    basisToUse = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    handLabeledReaches = []
    networkLabeledReaches = []
    patReaches = np.s_[:13]
    tonyReaches = np.s_[13:]
    reachPosThresh = -10
    
class DLC_params:
    fps = 180
    shortChunkLength = 200
    longGapLength = 20
    winSize = 51
    polyOrder = 3
    probWinSize = 21
    probPolyOrder = 3
    avgWin = 5
    handLabel = 2

#%% Load marker probabilities

windowSize = 51
polOrder = 3

prob_files_cam1 = sorted(glob.glob(params.traj_path + '*cam1*filtered*.h5'))
prob_files_cam2 = sorted(glob.glob(params.traj_path + '*cam2*filtered*.h5'))
cam1_prob = []
cam2_prob = []
for i in range(len(prob_files_cam1)):
    cam1_tmp = pd.read_hdf(prob_files_cam1[i])
    cam2_tmp = pd.read_hdf(prob_files_cam2[i])
    
    cam1_prob_tmp = cam1_tmp.iloc[:, 2::3]
    cam2_prob_tmp = cam2_tmp.iloc[:, 2::3]
    
    cam1_prob.append(savgol_filter(cam1_prob_tmp, windowSize, polOrder, axis = 0))
    cam2_prob.append(savgol_filter(cam2_prob_tmp, windowSize, polOrder, axis = 0))
    
#%% Compute DLC and XROMM basis matrices
    
dlc_ref_files = sorted(glob.glob(params.traj_path + '*ref*'))

def proj(u, v):   # Projection of u onto v
    return np.dot(u,v) / np.dot(v,v) * v

dlc_basis_mats = []
dlc_origin = []
for fNum, f in enumerate(dlc_ref_files):
    refPoints = np.load(f)
    refPoints = np.squeeze(refPoints)
    dlc_origin.append(refPoints[0, :])
    
    x = -1*(refPoints[1, :] - refPoints[0, :])
    x = x / np.linalg.norm(x)
    
    y = refPoints[2, :] - refPoints[1, :]
    y = y - proj(y, x)
    y = y / np.linalg.norm(y)
    
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    
    basis = np.column_stack((x, y, z))
    dlc_basis_mats.append(basis)  

#%% Load and process dlc trajectories (make sure to remove the statement 'if fNum != 4:' in this section for future use)
dlc_traj_files = sorted(glob.glob(params.traj_path + '*3D.h5')) 
dlc = []
for fNum, f in enumerate(dlc_traj_files):
    f = f.replace('\\', '/')
#    traj = np.load(f)
    traj = np.array(pd.read_hdf(f)).transpose() 
    traj = traj.reshape((int(np.shape(traj)[0] / 3), 3, np.shape(traj)[1]))

    traj = traj[0:3, :, :]

    for part in range(np.size(traj, 0)):
        for dim in range(3):      
            
            traj[part, dim, :] = np.squeeze(traj[part, dim, :]); 

            traj[part, :, cam1_prob[fNum][:, part] < 0.05] = np.nan
            traj[part, :, cam2_prob[fNum][:, part] < 0.05] = np.nan

        if np.nansum(~np.isnan(traj[part, ...])) > 0:
            traceIdxs = np.where(~np.isnan(traj[part, 0, :]))
            gapLength = np.diff(traceIdxs).flatten()
            bigGap_idxs = np.array(np.where(gapLength > DLC_params.longGapLength), dtype = int).flatten()
            bigGap_idxs = np.append(bigGap_idxs, np.max(traceIdxs))
            if np.min(bigGap_idxs) > 0:  ##### NOTE: I MIGHT NEED TO CHANGE THIS TO " > first notNaN index:"
                bigGap_idxs = np.insert(bigGap_idxs, 0, 0)
            
            storedGapIdxs = copy.copy(bigGap_idxs)
            
            chunkStarts = np.empty((np.size(bigGap_idxs) - 1, ), dtype = np.int16)
            chunkEnds = np.empty(np.shape(chunkStarts), dtype = np.int16)
            for i in range(len(chunkStarts)):
                tmpGaps = gapLength[:bigGap_idxs[i]+1]
                if bigGap_idxs[i] == 0:
                    chunkStarts[i] = 0
                else:
                    chunkStarts[i] = np.sum(tmpGaps)
            for i in range(len(chunkEnds)-1):
                chunkEnds[i] = chunkStarts[i+1] - gapLength[storedGapIdxs[i+1]] 
            chunkEnds[-1] = np.max(traceIdxs)
                            
            gapStarts = []
            gapEnds = []
            for i in range(len(chunkStarts)):
                mask = np.zeros((np.shape(traj)[-1]), dtype = bool)
                if chunkEnds[i] - chunkStarts[i] < DLC_params.shortChunkLength:
                    if fNum != 4:
                        if fNum == 7 and part == 3:
                            brekNow = []
                        mask[chunkStarts[i] : chunkEnds[i] + 1] = True
                        traj[part, :, mask] = np.nan
                    
            # Find remaining gaps and do a linear interpolation, followed by savgol_filter                
            traceIdxs = np.where(~np.isnan(traj[part, 0, :]))
            gapLength = np.diff(traceIdxs).flatten()
            gapIdxs = np.array(np.where(gapLength > 1), dtype = int).flatten()        
            storedGapIdxs = copy.copy(gapIdxs)
            
            if np.size(gapIdxs) > 0:
                gapStarts = np.empty((np.size(gapIdxs), ), dtype = np.int16)
                gapEnds = np.empty(np.shape(gapStarts), dtype = np.int16)
                for i in range(0, len(gapStarts)):
                    tmpGaps = gapLength[:gapIdxs[i]+1]
                    gapEnds[i] = gapIdxs[i] + np.sum(tmpGaps[tmpGaps > 1]) - i + traceIdxs[0][0]          
                    gapStarts[i] = gapEnds[i] - gapLength[gapIdxs[i]]
    
                    for dim in range(3):
                        tmpDiff = abs(np.diff(traj[part, dim, :]))
                        realDiff = tmpDiff[~np.isnan(tmpDiff)]
                        realDiff = abs(realDiff)
                        medDisp = np.median(realDiff)
                        
                        jumpCheck = 10
                        medMult = 10
                        bigJumpIdx = np.where(tmpDiff[gapEnds[i] : gapEnds[i] + jumpCheck] > medMult*medDisp)[0]
                        if np.shape(bigJumpIdx)[0] > 0:
                            traj[part, dim, gapEnds[i]:gapEnds[i] + bigJumpIdx[-1] + 1] = np.repeat(traj[part, dim, gapEnds[i] + bigJumpIdx[-1] + 1], bigJumpIdx[-1] + 1)                    
    
                        bigJumpIdx = np.where(tmpDiff[gapStarts[i] - 9 : gapStarts[i] + 1] > medMult*medDisp)[0]
                        if np.shape(bigJumpIdx)[0] > 0:
                            traj[part, dim, gapStarts[i] - (jumpCheck - bigJumpIdx[0]) + 1 : gapStarts[i] + 1 ] = np.repeat(traj[part, dim, gapStarts[i] - (jumpCheck - bigJumpIdx[0])], jumpCheck - bigJumpIdx[0])
                            
                        traj[part, dim, gapStarts[i]:gapEnds[i]+1] = np.linspace(traj[part, dim, gapStarts[i]], traj[part, dim, gapEnds[i]], gapEnds[i] - gapStarts[i]+1)   
    
        #### Comment this section to see unsmoothed data
            for dim in range(3):
                if sum(~np.isnan(traj[part, dim, :])) > DLC_params.winSize:
                    traj[part, dim, ~np.isnan(traj[part, dim, :])] = savgol_filter(traj[part, dim, ~np.isnan(traj[part, dim, :]).flatten()], DLC_params.winSize, DLC_params.polyOrder)        
        ####  
            
            cam1Prob_diff = savgol_filter(np.diff(cam1_prob[fNum][:, part]), DLC_params.probWinSize, DLC_params.probPolyOrder)
            cam2Prob_diff = savgol_filter(np.diff(cam2_prob[fNum][:, part]), DLC_params.probWinSize, DLC_params.probPolyOrder)
            
            cam1_mins = argrelextrema(cam1Prob_diff, np.less, order = 5)[0]
            cam2_mins = argrelextrema(cam2Prob_diff, np.less, order = 5)[0]
    
            cam1_maxs = argrelextrema(cam1Prob_diff, np.greater, order = 5)[0]
            cam2_maxs = argrelextrema(cam2Prob_diff, np.greater, order = 5)[0]
                    
            quickProbDrops = np.union1d(cam1_mins[cam1Prob_diff[cam1_mins] < -0.03], cam2_mins[cam2Prob_diff[cam2_mins] < -0.03])
            quickProbJumps = np.union1d(cam1_maxs[cam1Prob_diff[cam1_maxs] > 0.03], cam2_maxs[cam2Prob_diff[cam2_maxs] > 0.03])
            
            forwardProbAvg1 = np.empty(np.shape(np.squeeze(traj[part, 0, :])))
            forwardProbAvg2 = np.empty(np.shape(np.squeeze(traj[part, 0, :])))
            for idx in range(np.shape(traj)[-1]):
                distToEnd = np.shape(traj)[-1] - idx
                if distToEnd > DLC_params.avgWin:
                    forwardProbAvg1[idx] = np.nanmean(cam1_prob[fNum][idx : idx + DLC_params.avgWin, part]) 
                    forwardProbAvg2[idx] = np.nanmean(cam2_prob[fNum][idx : idx + DLC_params.avgWin, part]) 
                else:
                    forwardProbAvg1[idx] = np.nanmean(cam1_prob[fNum][idx : idx + distToEnd, part]) 
                    forwardProbAvg2[idx] = np.nanmean(cam2_prob[fNum][idx : idx + distToEnd, part])            
                   
            backwardProbAvg1 = np.empty(np.shape(np.squeeze(traj[part, 0, :])))
            backwardProbAvg2 = np.empty(np.shape(np.squeeze(traj[part, 0, :])))
            for idx in range(np.shape(traj)[-1]):
                if idx > DLC_params.avgWin:
                    backwardProbAvg1[idx] = np.nanmean(cam1_prob[fNum][idx - DLC_params.avgWin : idx + 1, part]) 
                    backwardProbAvg2[idx] = np.nanmean(cam2_prob[fNum][idx - DLC_params.avgWin : idx + 1, part]) 
                else:
                    backwardProbAvg1[idx] = np.nanmean(cam1_prob[fNum][:idx + 1, part]) 
                    backwardProbAvg2[idx] = np.nanmean(cam2_prob[fNum][:idx + 1, part])
                
            for idx in quickProbJumps:
                if sum(~np.isnan(traj[part, 0, :idx])) > 0 and sum(cam1_prob[fNum][:idx, part] >= 0.9) < DLC_params.shortChunkLength and sum(cam2_prob[fNum][:idx, part] >= 0.9) < DLC_params.shortChunkLength:
                    firstGoodPoint = np.intersect1d(np.where(forwardProbAvg1[idx:] >= 0.9)[0], np.where(forwardProbAvg2[idx:] >= 0.9)[0])
                    if len(firstGoodPoint) > 0:
                        traj[part, :, :firstGoodPoint[0] + idx] = np.nan
            for idx in quickProbDrops:
                if ~np.isnan(traj[part, 0, idx]) and sum(~np.isnan(traj[part, 0, idx:])) > 0 and sum(cam1_prob[fNum][idx:, part] >= .9) < DLC_params.shortChunkLength and sum(cam2_prob[fNum][idx:, part] >= .9) < DLC_params.shortChunkLength:
                    lastGoodPoint = np.intersect1d(np.where(backwardProbAvg1[:idx] >= 0.9)[0], np.where(backwardProbAvg2[:idx] >= 0.9)[0])
                    if len(lastGoodPoint) > 0:
                        traj[part, :, lastGoodPoint[-1]:] = np.nan
            
            
                    
#            traj[part, :, :] = traj[part, :, :] - np.repeat(dlc_origin[params.basisToUse[fNum]].reshape((3, 1)), np.shape(traj)[-1], axis = 1)
#            traj[part, :, :] = np.dot(dlc_basis_mats[params.basisToUse[fNum]], traj[part, :, :])
    
#    traj = traj[:, (2, 0, 1), :]
#    traj[:, 0, :] = -1 * traj[:, 0, :]
#    traj[:, 1, :] = -1 * traj[:, 1, :]
            
    traj = traj[:, (2, 0, 1), :]
            
    dlc.append(traj)

#%% Find points that are likely to be well-labeled by removing frames in which the hand is well behind the partition. Also remove non-overlapping points and mean-subtract. 

dlcVel = []
for fNum in range(len(dlc)):
    dlcTraj = dlc[fNum]
    
    nonMovePoints = np.where(dlcTraj[0, 1, :] < params.reachPosThresh)
    for part in range(np.size(dlcTraj, 0)):
        for dim in range(3):
            dlcTraj[part, dim, nonMovePoints] = np.nan
            remSlice = np.where(~np.isnan(dlcTraj[part, dim, :]))
            dlcTraj[part, dim, remSlice] = dlcTraj[part, dim, remSlice] - np.repeat(np.mean(dlcTraj[part, dim, remSlice]), len(remSlice))
        
    dlcVel_tmp = np.empty((np.shape(dlcTraj)[0], np.shape(dlcTraj)[1], np.shape(dlcTraj)[2] - 1))
    dlcTime = np.linspace(0, np.shape(dlcTraj)[2] / DLC_params.fps, num = np.shape(dlcTraj)[2])
    for part in range(np.size(dlcTraj, 0)):
        dlcVel_tmp[part, :, :] = np.divide(np.diff(dlcTraj[part, :, :], axis = 1), np.repeat(np.diff(dlcTime).reshape((1, len(dlcTime) - 1)), np.shape(dlcTraj)[1], axis = 0))

    speed = np.linalg.norm(dlcVel_tmp[DLC_params.handLabel, :, :], axis = 0)
    if np.nansum(~np.isnan(speed)) >= 51:
        speed[~np.isnan(speed)] = savgol_filter(speed[~np.isnan(speed)], 51, 3)
    
    restIdx = np.where(speed < 1)
    speed[restIdx] = np.nan

    minLen = 75

    gaps = np.where(np.isnan(speed))[0]
    gaps = np.insert(gaps, 0, 0)
    gapLengths = np.diff(gaps)
    gapsToRemove = np.vstack((gaps[:-1][np.logical_and(gapLengths > 1, gapLengths < minLen)], gaps[:-1][np.logical_and(gapLengths > 1, gapLengths < minLen)] + gapLengths[np.logical_and(gapLengths > 1, gapLengths < minLen)] + 1))
    gapsToRemove = gapsToRemove.transpose()
    
    gapsToFill = np.vstack(((gaps[:-1][gapLengths >= minLen] + gapLengths[gapLengths >= minLen] + 1)[:-1] - 1, gaps[:-1][gapLengths >= minLen][1:] + 1))
    gapsToFill = gapsToFill.transpose()
    
    for gap in gapsToRemove:
        speed[gap[0] : gap[1]] = np.nan
    
    for fill in gapsToFill:
        if fill[1] - fill[0] < 150:
            speed[fill[0] : fill[1]] = 0
            
    speed[-1] = np.nan

    plt.plot(dlcVel_tmp[2, 0, :] + 25, '-b')
    plt.plot(dlcVel_tmp[2, 1, :], '-g')
    plt.plot(dlcVel_tmp[2, 2, :] - 25, '-k')
    plt.show()

    speedTmp = np.append(speed, np.nan)
    dlcTraj[..., np.isnan(speedTmp)] = np.nan
    dlcVel_tmp[..., np.isnan(speed)] = np.nan
    
    plt.plot(dlcVel_tmp[2, 0, :] + 25, '-b')
    plt.plot(dlcVel_tmp[2, 1, :], '-g')
    plt.plot(dlcVel_tmp[2, 2, :] - 25, '-k')
    plt.show()

    dlcVel.append(dlcVel_tmp)
    dlc[fNum] = dlcTraj   

#%% Store data

trajectories = {'position': dlc, 'velocity': dlcVel}

dlc4mat = np.empty((len(dlc),), dtype=np.object)
dlcVel4mat = np.empty_like(dlc4mat)
for i in range(len(dlc)):
    dlc4mat[i]    = dlc[i]
    dlcVel4mat[i] = dlcVel[i]
trajectories4mat = {'position': dlc4mat, 'velocity': dlcVel4mat}

if operSystem == 'windows':  
    with open(params.traj_processedPath + '.p', 'wb') as f:
        pickle.dump(trajectories, f, protocol = pickle.HIGHEST_PROTOCOL)  
    savemat(params.traj_processedPath + '.mat', mdict = trajectories4mat)
elif operSystem == 'linux':  
    with open(params.tmpStorage + '.p', 'wb') as f:
        pickle.dump(trajectories, f, protocol = pickle.HIGHEST_PROTOCOL)  
    savemat(params.tmpStorage + '.mat', mdict = trajectories4mat)
    subprocess.run(['sudo', 'mv', params.tmpStorage+'.p', params.tmpStorage+'.mat', params.traj_processedPath]) 

#%% Plot individual sample trials

trajNum = 20
part = 2
time = np.linspace(0, np.shape(dlc[trajNum])[2] / DLC_params.fps * 1e3, num = np.shape(dlc[trajNum])[2])

plt.style.use('seaborn-whitegrid')

#plt.plot(dlc[trajNum][part, 2, :])
#plt.plot(xromm[trajNum][part, 2, :])
#plt.show()

plt.plot(time, dlc[trajNum][part, 0, :] + 5, '-b')
plt.plot(time, dlc[trajNum][part, 1, :], '-g')
plt.plot(time, dlc[trajNum][part, 2, :] - 5, '-k')

plt.show()

