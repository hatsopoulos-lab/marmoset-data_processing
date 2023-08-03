import numpy as np
import pandas as pd
import seaborn as sns
import glob
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import matplotlib
import copy
import warnings
from statsmodels.stats.weightstats import DescrStatsW
import mpl_toolkits.mplot3d.axes3d as p3

downSampRate = 200    
    
    # Pat_15
        # 002 (0)  ++ - DONE - super short reach                                
        # 003**    ++ - DONE - two reaches, good
        # 007      ++ - DONE - one reach, good, small errors in various body parts but fine
        # 008      ++ - DONE - forearm proximal jumps briefly in cam1, upperArm medial extends past time of other markers --> check code
        # 013**    ++ - NOT DONE - very short reach (need to check why reach is cut off too soon)
        # 015**    ++ - DONE - medium reach, most markers good. Torso-upper should cut off sooner --> check code. Forearm doesn't track sup/pronation well
        # 018      ++ - DONE - 2 long reaches, great
    # Pat_14
        # 001 (7)  ++ - DONE - torso obscured by Tony
        # 022**    ++ - NOT DONE - everything cuts off before reach
        # 023      -- - NOT DONE - everything cuts off beginning and end of reach (cam2-415, 475, 710-800)(cam1-415,475,750-800)
        # 026**    -- - NOT DONE - Hand very low prob
        # 046**    ++ - DONE
        # 048      ++ - DONE - good
    # Tony_14
        # 013 (13) ++ - DONE - very short reach, good. However, upperArm proximal extends longer in time than rest of markers
        # 015      ++ - DONE - Pat interferes and obscures torso and upperArm Distal/proximal earlier than other parts
        # 021      ++ - NOT DONE - Check torso back labeling and forearm proximal
        # 028      ++ - DONE
        
class params:
    # traj_path = r'Z:/marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/corrected_calibration_post_first_refinement/'
    traj_path = r'Z:/marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/post_first_refinement_copy_with_removedTrajectories/'
    x_traj_path = r'Z:/marmosets/XROMM_and_RGB_sessions/XROMM_videos/validation_trajectories/'   
    anipose = False     
    sortedFiles_reorder = [8, 9, 10, 11, 12, 13, 14, 0, 4, 6, 7, 1, 2, 3, 5]
    basisToUse = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    handLabeledReaches = [0, 2, 3, 6, 7, 10, 11, 12, 13, 14]
    networkLabeledReaches = [1, 4, 5, 8, 9]
    patReaches = np.s_[:11]
    tonyReaches = np.s_[11:]
    reachPosThresh = -1.5
    saveFigPath = r'C:/Users/Dalton/Documents/lab_files/dlc_validation_paper/figures_tmp/'

class XROMM_params:
    fps = 200
    downSampFraction = downSampRate / fps
    labelOrder = [12, 11, 10, 9, 8, 7, 6, 5, 3, 4, 2, 1]
    winSize = round(31 * downSampFraction)
    if winSize % 2 == 0:
        winSize += 1
    polyOrder = 3

class DLC_params:
    fps = 200
    downSampFraction = downSampRate / fps
    likeThresh = 0.4
    looseLikeThresh = 0.05
    labelOrder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    shortChunkLength = round(200 * downSampFraction)
    longGapLength = round(20 * downSampFraction)
    gapTooBigToFill = round(300 * downSampFraction)
    winSize = round(31 * downSampFraction)
    if winSize % 2 == 0:
        winSize += 1
    polyOrder = 3
    probWinSize = round(21 * downSampFraction)
    if probWinSize % 2 == 0:
        probWinSize += 1
    probPolyOrder = 3
    avgWin = round(5 * downSampFraction)

#### Need to skip some of the processing chunks that are already done by anipose, if anipose

#%% Load marker probabilities

if not params.anipose:

    windowSize = round(51 * DLC_params.downSampFraction)
    if windowSize % 2 == 0:
        windowSize += 1
    polOrder = 3
    
    prob_files_cam1 = sorted(glob.glob(params.traj_path + '*cam1*.h5'))
    prob_files_cam2 = sorted(glob.glob(params.traj_path + '*cam2*.h5'))
    
    prob_files_cam1 = [prob_files_cam1[i] for i in params.sortedFiles_reorder]
    prob_files_cam2 = [prob_files_cam2[i] for i in params.sortedFiles_reorder]
        # figure out rotation with extra labels to get into same coordinate frame
    cam1_prob = []
    cam2_prob = []
    for i in range(len(prob_files_cam1)):
        cam1_tmp = pd.read_hdf(prob_files_cam1[i])
        cam2_tmp = pd.read_hdf(prob_files_cam2[i])
        
        cam1_prob_tmp = cam1_tmp.iloc[:, 2::3]
        cam2_prob_tmp = cam2_tmp.iloc[:, 2::3]
        
        cam1_prob_tmp = savgol_filter(cam1_prob_tmp, windowSize, polOrder, axis = 0)
        cam2_prob_tmp = savgol_filter(cam2_prob_tmp, windowSize, polOrder, axis = 0)
        
        cam1_prob.append(cam1_prob_tmp)
        cam2_prob.append(cam2_prob_tmp)
        
        # if i == 8:
        #     time = np.linspace(0, len(cam1_prob_tmp) / 200 * 1000, len(cam1_prob_tmp))
        #     plt.plot(time, cam1_prob_tmp[:, 0])
        #     plt.plot(time, cam2_prob_tmp[:, 0])
        #     # plt.plot(cam1_prob_tmp[:, 0])
        #     # plt.plot(cam2_prob_tmp[:, 0])
        #     plt.show()
        
#%% Compute DLC and XROMM basis matrices
    
dlc_ref_files = sorted(glob.glob(params.traj_path + '*refFrame*'))
xromm_ref_files = sorted(glob.glob(params.x_traj_path + '*axes*')) 

def proj(u, v):   # Projection of u onto v
    return np.dot(u,v) / np.dot(v,v) * v

# the final (x,y,z) = (-z, -x, y). For some reason organizing it this way now resuts in errors, 
# while changing the order after processing works correctly. No idea why, 
# but I decided not to worry about it and change the order/direction later on at the end of each processing block.

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
    
    # if fNum == 1:
    #     img1 = cv2.imread(r'C:/Users/daltonm/Documents/Lab_Files/updated_trajectory_analysis/trajectories_code_and_figures/DLC_validation/origin_cam1.png')
    #     img2 = cv2.imread(r'C:/Users/daltonm/Documents/Lab_Files/updated_trajectory_analysis/trajectories_code_and_figures/DLC_validation/origin_cam2.png')
    #     tmpImg1 = np.copy(img1)
    #     originPoint = refPoints[0, :]
    #     xPoint = originPoint - 5*z
    #     yPoint = originPoint - 5*x
    #     zPoint = originPoint + 5*y

    #     cam1_rvec      = np.array([ 0.76214443, -0.3577923, 0.06610537]).astype(np.float32)
    #     cam1_tvec      = np.array([-3.33566974, 2.90972961, 35.95017098]).astype(np.float32)

    #     # cam1_rvec      = np.array([0, 0, 0]).astype(np.float32)
    #     # cam1_tvec      = np.array([0, 0, 0]).astype(np.float32)

    #     cam2_rvec      = np.array([0, 0, 0]).astype(np.float32)
    #     cam2_tvec      = np.array([0, 0, 0]).astype(np.float32)

    #     cam1_mat       = np.array([[2408.31086, 0         , 744.606275],
    #                                [0         , 2410.48885, 521.416025],
    #                                [0         , 0         , 1         ]])
    #     cam2_mat       = np.array([[1618.88248, 0         , 787.789690],
    #                                [0         , 1604.08945, 556.236683],
    #                                [0         , 0         , 1         ]])
    #     cam1_distCoeff = np.array([-0.295935490, -.682596831, -0.00183053344, -0.00195633780, -2.45871880])
    #     # cam1_distCoeff = np.array([0, 0, 0, 0, 0]).astype(np.float32)

    #     cam2_distCoeff = np.array([-0.342793152, -.551808998,  0.00208599636, -0.00220794010,  5.36956072])
        
    #     cam1_points, tmp = cv2.projectPoints(np.vstack((originPoint, xPoint, yPoint, zPoint)), cam1_rvec, cam1_tvec, cam1_mat, cam1_distCoeff)
    #     cam2_points, tmp = cv2.projectPoints(np.vstack((originPoint, xPoint, yPoint, zPoint)), cam2_rvec, cam2_tvec, cam2_mat, cam2_distCoeff)                
    #     cam1_points = cam1_points.squeeze().astype(np.int32)
    #     cam2_points = cam2_points.squeeze().astype(np.int32)
    #     colors = [tuple([int(255), int(255), int(0)]),
    #              tuple([int(255), int(0), int(0)]),
    #              tuple([int(0), int(0), int(255)])]
    #     for pts1, pts2, color in zip(cam1_points[1:, :], cam2_points[1:, :], colors):
    #         img1 = cv2.line(img1, (cam1_points[0,0],cam1_points[0,1]), (pts1[0], pts1[1]), color, 5)
    #         img2 = cv2.line(img2, (cam2_points[0,0],cam2_points[0,1]), (pts2[0], pts2[1]), color, 5)
    #     plt.figure()
    #     plt.imshow(img1)
    #     plt.show()
    #     # plt.figure()
    #     # plt.imshow(img2)

    #     # plt.show()
  
xromm_basis_mats = []
xromm_origin = []    
for fNum, f in enumerate(xromm_ref_files):
    refPoints = np.loadtxt(f, delimiter = ',', skiprows = 1)
    refPoints = refPoints[~np.isnan(refPoints[:, 0]), :]
    refPoints = np.reshape(refPoints, (3, 3))
    xromm_origin.append(refPoints[0, :])
    
    x = -1*(refPoints[1, :] - refPoints[0, :])
    x = x / np.linalg.norm(x)
    
    y = refPoints[2, :] - refPoints[1, :]
    y = y - proj(y, x)
    y = y / np.linalg.norm(y)
    
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    
    basis = np.column_stack((x, y, z))
    xromm_basis_mats.append(basis)

#%% Load and process dlc trajectories (make sure to remove the statement 'if fNum != 4:' in this section for future use)
print('\n Loading DLC trajectories...')

dlc_traj_files = sorted(glob.glob(params.traj_path + '*15*Pat*traj*')) 
dlc_traj_files.extend(sorted(glob.glob(params.traj_path + '*14*Pat*traj*')))
dlc_traj_files.extend(sorted(glob.glob(params.traj_path + '*14*Tony*traj*')))
dlc = []
unconnectedDLC = []
unsmoothedDLC = []
unprocessedDLC = []
for fNum, f in enumerate(dlc_traj_files):
    
    print('\n Processing DLC trajectory ' + str(fNum+1) + ' of ' + str(len(dlc_traj_files)))
    f = f.replace('\\', '/')
    traj = np.load(f)
    
    
    if not params.anipose:

        unprocessedTraj = traj.copy()
    
        unconnectedTraj = np.empty_like(traj)
        unsmoothedTraj = np.empty_like(traj)
        for part in range(np.size(traj, 0)):
            
            # filter out frames with marker likelihood below the threshold (defined above)
            if fNum == 12:
                cam1_filter = np.where(cam1_prob[fNum][:, part] < DLC_params.likeThresh)[0]
                cam2_filter = np.where(cam2_prob[fNum][:, part] < DLC_params.likeThresh)[0]
                cam1_filter = [i for i in cam1_filter if i < 795 or i > 814]
                cam2_filter = [i for i in cam2_filter if i < 795 or i > 814]
                traj[part, :, cam1_filter] = np.nan
                traj[part, :, cam2_filter] = np.nan
            elif fNum == 8:
                traj[part, :, cam1_prob[fNum][:, part] < DLC_params.looseLikeThresh] = np.nan
                traj[part, :, cam2_prob[fNum][:, part] < DLC_params.looseLikeThresh] = np.nan
            else: 
                traj[part, :, cam1_prob[fNum][:, part] < DLC_params.likeThresh] = np.nan
                traj[part, :, cam2_prob[fNum][:, part] < DLC_params.likeThresh] = np.nan
    
            # find big gaps between remaining chunks. Gaps smaller than DLC_params.longGapLength are treated as 
            # brief blips and ignored for further processing --> will be filled later     
            traceIdxs = np.where(~np.isnan(traj[part, 0, :]))
            gapLength = np.diff(traceIdxs).flatten()
            bigGap_idxs = np.array(np.where(gapLength > DLC_params.longGapLength), dtype = int).flatten()
            bigGap_idxs = np.append(bigGap_idxs, np.max(traceIdxs))
            if np.min(bigGap_idxs) > 0:  ##### NOTE: I MIGHT NEED TO CHANGE THIS TO " > first notNaN index:"
                bigGap_idxs = np.insert(bigGap_idxs, 0, 0)
            
            # use the identified gaps to identify the start and end of remaining chunks
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
            
            # remove chunks that are shorter than DLC_params.shortChunkLength. This is meant to get rid of
            # short chunks of high-likelihood tracking that are likely to be insignificant and should not be 
            # connected to the remaining chunks. If epochs of movement that should be retained are eliminated here,
            # the labeling of data videos needs to be improved thru the DLC refinement toolbox.
                     
            for i in range(len(chunkStarts)):
                mask = np.zeros((np.shape(traj)[-1]), dtype = bool)
                if chunkEnds[i] - chunkStarts[i] < DLC_params.shortChunkLength:
                    if fNum != 4:
                        mask[chunkStarts[i] : chunkEnds[i] + 1] = True
                        traj[part, :, mask] = np.nan
     
            # Find remaining gaps and do a linear interpolation to cover the gaps between chunks 
            # that were retained aboved, followed by savgol_filter                
            traceIdxs = np.where(~np.isnan(traj[part, 0, :]))
            gapLength = np.diff(traceIdxs).flatten()
            gapIdxs = np.array(np.where(gapLength > 1), dtype = int).flatten()        
            storedGapIdxs = copy.copy(gapIdxs)
    
            # unconnectedTraj[part, :, :] = traj[part, :, :]
            
            if np.size(gapIdxs) > 0:
                gapStarts = np.empty((np.size(gapIdxs), ), dtype = np.int16)
                gapEnds = np.empty(np.shape(gapStarts), dtype = np.int16)
                for i in range(0, len(gapStarts)):
                    # find start and end of gaps between chunks
                    tmpGaps = gapLength[:gapIdxs[i]+1]
                    gapEnds[i] = gapIdxs[i] + np.sum(tmpGaps[tmpGaps > 1]) - i + traceIdxs[0][0]          
                    gapStarts[i] = gapEnds[i] - gapLength[gapIdxs[i]]
    
                    for dim in range(3):
                        # before connecting chunks across the gaps, check if there are large 
                        # marker movements (> 10 times the median frame to frame displacement of the event) 
                        # within 10 frames of the edges of the chunks. If there are, replace the frames  
                        # of the large movements with the first (or last) good frame of the chunk. 
                        tmpDiff = abs(np.diff(traj[part, dim, :]))
                        realDiff = tmpDiff[~np.isnan(tmpDiff)]
                        realDiff = abs(realDiff)
                        medDisplacement = np.median(realDiff)
    
                        jumpCheck = 15
                        medMult = 10
                        
                        # check the beginning of each chunk (end of gap)
                        jumpCheck_idx = [idx for idx in range(gapEnds[i], gapEnds[i] + jumpCheck) if idx < len(tmpDiff) and ~np.isnan(tmpDiff[idx])]
                        bigJumpIdx = np.where(abs(tmpDiff[jumpCheck_idx]) > medMult*medDisplacement)[0]
                        if np.shape(bigJumpIdx)[0] > 0:
                            # traj[part, dim, gapEnds[i]:gapEnds[i] + bigJumpIdx[-1] + 1] = np.repeat(traj[part, dim, gapEnds[i] + bigJumpIdx[-1] + 1], bigJumpIdx[-1] + 1)                    
                            traj[part, dim, gapEnds[i]:gapEnds[i] + bigJumpIdx[-1] + 1] = np.nan
                            gapEnds[i] = gapEnds[i] + bigJumpIdx[-1]
    
                        # check the end of each chunk (beginning of gap)
                        jumpCheck_idx = [idx for idx in range(gapStarts[i] - (jumpCheck-1), gapStarts[i] + 1) if idx >= 0 and ~np.isnan(tmpDiff[idx])]
                        bigJumpIdx = np.where(abs(tmpDiff[jumpCheck_idx]) > medMult*medDisplacement)[0]
                        if np.shape(bigJumpIdx)[0] > 0:
                            # traj[part, dim, gapStarts[i] - (jumpCheck - bigJumpIdx[0]) + 1 : gapStarts[i] + 1 ] = np.repeat(traj[part, dim, gapStarts[i] - (jumpCheck - bigJumpIdx[0])], jumpCheck - bigJumpIdx[0])
                            traj[part, dim, gapStarts[i] - (jumpCheck - bigJumpIdx[0]) + 1 : gapStarts[i] + 1 ] = np.nan
                            gapStarts[i] = gapStarts[i] - (jumpCheck - bigJumpIdx[0]) + 1
                            
                        if gapStarts[i] < 0:
                            gapStarts[i] = 0
                        if gapEnds[i] > len(traj[part, dim, :]):
                            gapEnds[i] = len(traj[part, dim, :])
                            
                        if gapEnds[i] - gapStarts[i] < DLC_params.gapTooBigToFill:
                            traj[part, dim, gapStarts[i]:gapEnds[i]+1] = np.linspace(traj[part, dim, gapStarts[i]], traj[part, dim, gapEnds[i]], gapEnds[i] - gapStarts[i]+1)   

            # identify very brief marker jumps (a few frames)       
            for dim in range(3):                   
                diffFrame2Frame = np.diff(traj[part, dim, :])
                nanIdx = []
                for idx, dx in enumerate(diffFrame2Frame):
                    if abs(dx) > 0.75 and abs(dx) > np.nanmedian(abs(diffFrame2Frame[idx-3 : idx+4]))*8:
                        nanIdx.append(idx)
                        # print(idx)
                moreNanIdx = []
                for firstIdx, secondIdx in zip(nanIdx[:-1], nanIdx[1:]):
                    space = secondIdx - firstIdx
                    if space <= 3:
                        moreNanIdx.extend(list(range(firstIdx+1, secondIdx)))
                        moreNanIdx.extend(list(range(secondIdx+1, secondIdx+space)))
                allNanIdx = np.union1d(nanIdx, moreNanIdx).astype(np.int16)
                    
                traj[part, dim, allNanIdx] = np.nan
                        
                traceIdxs = np.where(~np.isnan(traj[part, dim, :]))
                gapLength = np.diff(traceIdxs).flatten()
                gapIdxs = np.array(np.where(gapLength > 1), dtype = int).flatten()

                if np.size(gapIdxs) > 0:
                    for i in range(0, len(gapIdxs)):
                        # find start and end of gaps between chunks
                        tmpGaps = gapLength[:gapIdxs[i]+1]
                        gapEnd = int(gapIdxs[i] + np.sum(tmpGaps[tmpGaps > 1]) - i + traceIdxs[0][0])     
                        gapStart = int(gapEnd - gapLength[gapIdxs[i]])
    
                        if gapEnd - gapStart < DLC_params.gapTooBigToFill:
                            traj[part, dim, gapStart:gapEnd+1] = np.linspace(traj[part, dim, gapStart], traj[part, dim, gapEnd], gapEnd - gapStart+1)

            unconnectedTraj[part, :, :] = traj[part, :, :] #+++
    
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
                
            if ~np.logical_and(fNum == 4, part in [6, 7, 8, 9]):        
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
                
            traceIdxs = np.where(~np.isnan(traj[part, 0, :]))
            if np.size(traceIdxs[0]) > 0:
                gapLength = np.diff(traceIdxs).flatten()
                bigGap_idxs = np.array(np.where(gapLength > 1), dtype = int).flatten()
                bigGap_idxs = np.append(bigGap_idxs, np.max(traceIdxs))
                if np.min(bigGap_idxs) > 0:  ##### NOTE: I MIGHT NEED TO CHANGE THIS TO " > first notNaN index:"
                    bigGap_idxs = np.insert(bigGap_idxs, 0, 0)
                
                # use the identified gaps to identify the start and end of remaining chunks
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
                
                for start, stop in zip(chunkStarts, chunkEnds):
                    for dim in range(3):
                        frameDiff = np.diff(traj[part, dim, :])
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', category=RuntimeWarning)
                            bigJump = np.where(abs(frameDiff) > 1.5)[0]
                            
                        for idx in bigJump:
                            if idx - start < 125 and idx-start < stop-idx:
                                traj[part, :, :idx+5] = np.nan
                            elif stop - idx < 125 and stop-idx<idx-start:
                                traj[part, :, idx-4:] = np.nan

            unsmoothedTraj[part, :, :] = traj[part, :, :]  #+++  
    
        #### Comment this section to see unsmoothed data
            for dim in range(3):
                if sum(~np.isnan(traj[part, dim, :])) > DLC_params.winSize:
                    traj[part, dim, ~np.isnan(traj[part, dim, :])] = savgol_filter(traj[part, dim, ~np.isnan(traj[part, dim, :]).flatten()], DLC_params.winSize, DLC_params.polyOrder)        
        #### 
                
            traj[part, :, :] = traj[part, :, :] - np.repeat(dlc_origin[params.basisToUse[fNum]].reshape((3, 1)), np.shape(traj)[-1], axis = 1)
            traj[part, :, :] = np.dot(dlc_basis_mats[params.basisToUse[fNum]], traj[part, :, :])
    
            unconnectedTraj[part, :, :] = unconnectedTraj[part, :, :] - np.repeat(dlc_origin[params.basisToUse[fNum]].reshape((3, 1)), np.shape(unconnectedTraj)[-1], axis = 1)
            unconnectedTraj[part, :, :] = np.dot(dlc_basis_mats[params.basisToUse[fNum]], unconnectedTraj[part, :, :])
    
            unsmoothedTraj[part, :, :] = unsmoothedTraj[part, :, :] - np.repeat(dlc_origin[params.basisToUse[fNum]].reshape((3, 1)), np.shape(unsmoothedTraj)[-1], axis = 1)
            unsmoothedTraj[part, :, :] = np.dot(dlc_basis_mats[params.basisToUse[fNum]], unsmoothedTraj[part, :, :])
    
            unprocessedTraj[part, :, :] = unprocessedTraj[part, :, :] - np.repeat(dlc_origin[params.basisToUse[fNum]].reshape((3, 1)), np.shape(unprocessedTraj)[-1], axis = 1)
            unprocessedTraj[part, :, :] = np.dot(dlc_basis_mats[params.basisToUse[fNum]], unprocessedTraj[part, :, :])      

    elif params.anipose:
        for part in range(traj.shape[0]):
            traj[part, :, :] = traj[part, :, :] - np.repeat(dlc_origin[params.basisToUse[fNum]].reshape((3, 1)), np.shape(traj)[-1], axis = 1)
            traj[part, :, :] = np.dot(dlc_basis_mats[params.basisToUse[fNum]], traj[part, :, :])        

    traj = traj[:, (2, 0, 1), :]
    traj[:, 0, :] = -1 * traj[:, 0, :]
    traj[:, 1, :] = -1 * traj[:, 1, :]

    dlc.append(traj)

    if not params.anipose:

        unconnectedTraj = unconnectedTraj[:, (2, 0, 1), :]
        unconnectedTraj[:, 0, :] = -1 * unconnectedTraj[:, 0, :]
        unconnectedTraj[:, 1, :] = -1 * unconnectedTraj[:, 1, :]
    
        unsmoothedTraj = unsmoothedTraj[:, (2, 0, 1), :]
        unsmoothedTraj[:, 0, :] = -1 * unsmoothedTraj[:, 0, :]
        unsmoothedTraj[:, 1, :] = -1 * unsmoothedTraj[:, 1, :]
    
        unprocessedTraj = unprocessedTraj[:, (2, 0, 1), :]
        unprocessedTraj[:, 0, :] = -1 * unprocessedTraj[:, 0, :]
        unprocessedTraj[:, 1, :] = -1 * unprocessedTraj[:, 1, :]
                
        unconnectedDLC.append(unconnectedTraj)
        unsmoothedDLC.append(unsmoothedTraj)
        unprocessedDLC.append(unprocessedTraj)

#%% Load XROMM trajectories

print('\n Loading XROMM trajectories')

xromm_traj_files = sorted(glob.glob(params.x_traj_path + '*event*.csv'))
xromm_traj_files.pop(6)
xromm_traj_files.pop(5)
xromm_traj_files = [xromm_traj_files[i] for i in params.sortedFiles_reorder] 
xromm = []
for fNum, f in enumerate(xromm_traj_files):
    f = f.replace('\\', '/')
    tmp_traj = np.loadtxt(f, delimiter = ',', skiprows = 1)
    
    traj = np.empty((int(np.size(tmp_traj, 1) / 3), 3, np.size(tmp_traj, 0)), dtype=np.float64)
    for part in range(int(np.size(tmp_traj, 1) / 3)):
        traj[part, :, :] = tmp_traj[:, 3*part : 3*part+3].transpose()
        traj[part, :, :] = traj[part, :, :] - np.repeat(xromm_origin[params.basisToUse[fNum]].reshape((3, 1)), np.shape(traj)[-1], axis = 1)        
        traj[part, :, :] = np.dot(xromm_basis_mats[params.basisToUse[fNum]], traj[part, :, :])

    if np.shape(traj)[0] < 13:
        nanVec = np.full_like(traj[:(13 - np.shape(traj)[0]), :, :], np.nan)    
        traj = np.insert(traj, 0, nanVec, axis=0)
            
    traj = traj[XROMM_params.labelOrder, :, :]    
    traj = traj[:, (2, 0, 1), :]
    traj[:, 0, :] = -1 * traj[:, 0, :]
    traj[:, 1, :] = -1 * traj[:, 1, :]

    xromm.append(traj)
    
#%% Find points that are likely to be well-labeled by removing frames in which the hand is well behind the partition. Also remove non-overlapping points and mean-subtract. 

print('\n Further processing - 1) remove frames in which the hand is well behind the partition')
print('                      2) align DLC and XROMM trajectories')
print('                      3) calculate velocity traces')
print('                      4) remove large marker jumps based on velocity spikes')

dlcVel = []
xrommVel = []
storeDiff = []
bestShift = []
xromm_allData = []
xrommVel_allData = []
handCutoff_forPlot = []
percentageTracked = np.empty((len(dlc_traj_files), 12))
dlcMatchingLength = np.empty_like(percentageTracked)
xrommTrackingLength = np.empty_like(percentageTracked)
dlcTrajLength = np.empty_like(percentageTracked)
trackedXrommPoints = []
meanPositionSubtracted = []
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    for fNum in range(len(dlc)):
        dlcTraj = dlc[fNum]

        handNotDefinedFrames = np.where(np.isnan(dlcTraj[0, 1, :]))[0]
        nonMovePoints = np.where(dlcTraj[0, 1, :] < params.reachPosThresh)
        for part in range(np.size(dlcTraj, 0)):
            for dim in range(3):
                if fNum != 9:
                    dlcTraj[part, dim, nonMovePoints] = np.nan
                if fNum != 4:
                    dlcTraj[part, dim, handNotDefinedFrames] = np.nan
                            
        xrommTraj = xromm[fNum]
        goodXrommPoints = np.where(xrommTraj[0, 1, :] >= params.reachPosThresh)[0]
        trackedXrommPoints.append(goodXrommPoints[:-1])
    
        diff = abs(np.shape(dlcTraj)[-1] - np.shape(xrommTraj)[-1])
        if diff == 0:
            diff = 5
    
        lastError = 1000
        for delta in range(-diff, diff):
            dlcTraj_copy = dlcTraj.copy()
            xrommTraj_copy = xrommTraj.copy()
            unfiltered_xromm = xrommTraj.copy()
            if not params.anipose:
                unprocessedTraj_copy = unprocessedDLC[fNum].copy()
            if delta <= 0:
                start = np.zeros(np.shape(dlcTraj_copy[:, :, 0:abs(delta)]))
                end = np.zeros(np.shape(dlcTraj_copy[:, :, 0:diff - abs(delta)]))
                start[start == 0] = np.nan
                end[end == 0] = np.nan
                dlcTraj_copy = np.dstack((start, dlcTraj_copy, end))
            else:
                end = np.zeros(np.shape(dlcTraj_copy[:, :, 0:2*delta]))
                end[end == 0] = np.nan 
                dlcTraj_copy = np.dstack((dlcTraj_copy, end))
                dlcTraj_copy = np.delete(dlcTraj_copy, np.s_[0:delta], axis = 2)
    
            tmpDiff = np.shape(dlcTraj_copy)[-1] - np.shape(xrommTraj_copy)[-1]
            adjust = np.zeros(np.shape(dlcTraj_copy[:, :, 0: abs(tmpDiff) ]))
            adjust[adjust == 0] = np.nan
    
            if tmpDiff > 0:
                xrommTraj_copy = np.dstack((xrommTraj_copy, adjust))
                unfiltered_xromm = np.dstack((unfiltered_xromm, adjust))
            else:
                dlcTraj_copy = np.dstack((dlcTraj_copy, adjust))            
            
            rmse_tmp = np.empty(np.shape(dlcTraj_copy[:, :, 0]))  
            meanSubtract = np.empty((12, 3))
            for part in range(np.size(dlcTraj_copy, 0)):
                nonOverlap = np.where(np.logical_or(np.isnan(xrommTraj_copy[part, 0, :]), np.isnan(dlcTraj_copy[part, 0, :])))
                for dim in range(3):
                    dlcTraj_copy[part, dim, nonOverlap] = np.nan
                    xrommTraj_copy[part, dim, nonOverlap] = np.nan
                    
                    remSlice = np.where(~np.isnan(dlcTraj_copy[part, dim, :]))
                    if part == 0 and dim == 1:
                        tmpHandCutoff = params.reachPosThresh - np.mean(dlcTraj_copy[part, dim, remSlice])

                    meanSubtract[part, dim] = np.mean(dlcTraj_copy[part, dim, remSlice])
                    dlcTraj_copy[part, dim, remSlice] = dlcTraj_copy[part, dim, remSlice] - np.repeat(np.mean(dlcTraj_copy[part, dim, remSlice]), len(remSlice))
                    xrommTraj_copy[part, dim, remSlice] = xrommTraj_copy[part, dim, remSlice] - np.repeat(np.mean(xrommTraj_copy[part, dim, remSlice]), len(remSlice))
                    if len(remSlice[0]) > 0:
                        unfiltered_xromm[part, dim, :] = unfiltered_xromm[part, dim, :] - np.repeat(np.mean(unfiltered_xromm[part, dim, remSlice]), len(unfiltered_xromm[0,0,:]))
                error = xrommTraj_copy[part, :, :] - dlcTraj_copy[part, :, :]
                if np.any(~np.isnan(error)):
                    rmse_tmp[part, :] = np.sqrt(np.nanmean(np.square(error), axis = 1)).transpose()
                else:
                    rmse_tmp[part, :] = np.array([np.nan, np.nan, np.nan])
            
            newError = np.nanmean(rmse_tmp)
            if newError < lastError:
                lastError = newError
                diff_tmp = delta      
                shifted_dlcTraj_tmp = dlcTraj_copy
                shifted_xrommTraj_tmp = xrommTraj_copy
                shifted_unfiltered_xromm = unfiltered_xromm
                bestShiftHandCutoff = tmpHandCutoff
                bestShiftMean = meanSubtract
        
        if lastError == 1000:
            bestShift.append(0)
            diff = np.shape(dlcTraj)[-1] - np.shape(xrommTraj)[-1]
            end = np.zeros(np.shape(dlcTraj_copy[:, :, 0:abs(diff)]))
            end[end == 0] = np.nan
            if diff <= 0: 
                dlcTraj = np.dstack((dlcTraj, end))
            else:
                xrommTraj = np.dstack((xrommTraj, end))
            handCutoff_forPlot.append(params.reachPosThresh)
            allDataPointsXromm = xrommTraj
        else:
            bestShift.append(diff_tmp)
            handCutoff_forPlot.append(bestShiftHandCutoff)
            dlcTraj = shifted_dlcTraj_tmp
            xrommTraj = shifted_xrommTraj_tmp
            allDataPointsXromm = shifted_unfiltered_xromm
            
            if not params.anipose:
                for part in range(np.size(dlcTraj_copy, 0)):
                    for dim in range(3):
                        unprocessedDLC[fNum][part, dim, :] = unprocessedDLC[fNum][part, dim, :] - np.repeat(bestShiftMean[part, dim], len(unprocessedDLC[fNum][part, dim, :]))
        
        # compute velocity profiles
        dlcVel_tmp = np.empty((np.shape(dlcTraj)[0], np.shape(dlcTraj)[1], np.shape(dlcTraj)[2] - 1))
        xrommVel_tmp = np.empty((np.shape(xrommTraj)[0], np.shape(xrommTraj)[1], np.shape(xrommTraj)[2] - 1))
        allDataPointsXrommVel = np.empty((np.shape(allDataPointsXromm)[0], np.shape(allDataPointsXromm)[1], np.shape(allDataPointsXromm)[2] - 1))
        dlcTime = np.linspace(0, np.shape(dlcTraj)[2] / DLC_params.fps, num = np.shape(dlcTraj)[2])
        xrommTime = np.linspace(0, np.shape(xrommTraj)[2] / XROMM_params.fps, num = np.shape(xrommTraj)[2])
        for part in range(np.size(dlcTraj, 0)):
            dlcVel_tmp[part, :, :] = np.divide(np.diff(dlcTraj[part, :, :], axis = 1), np.repeat(np.diff(dlcTime).reshape((1, len(dlcTime) - 1)), np.shape(dlcTraj)[1], axis = 0))
            xrommVel_tmp[part, :, :] = np.divide(np.diff(xrommTraj[part, :, :], axis = 1), np.repeat(np.diff(xrommTime).reshape((1, len(xrommTime) - 1)), np.shape(xrommTraj)[1], axis = 0))
            allDataPointsXrommVel[part, :, :] = np.divide(np.diff(allDataPointsXromm[part, :, :], axis = 1), np.repeat(np.diff(xrommTime).reshape((1, len(xrommTime) - 1)), np.shape(xrommTraj)[1], axis = 0)) 
            for dim in range(3):
                if sum(~np.isnan(xrommVel_tmp[part, dim, :])) > XROMM_params.winSize:
                    xrommVel_tmp[part, dim, ~np.isnan(xrommVel_tmp[part, dim, :])] = savgol_filter(xrommVel_tmp[part, dim, ~np.isnan(xrommVel_tmp[part, dim, :]).flatten()], XROMM_params.winSize, XROMM_params.polyOrder)
                if sum(~np.isnan(allDataPointsXrommVel[part, dim, :])) > XROMM_params.winSize:
                    allDataPointsXrommVel[part, dim, ~np.isnan(allDataPointsXrommVel[part, dim, :])] = savgol_filter(allDataPointsXrommVel[part, dim, ~np.isnan(allDataPointsXrommVel[part, dim, :]).flatten()], XROMM_params.winSize, XROMM_params.polyOrder)
        
        # idenity marker jumps using velocity profiles and remove them
        for part in range(np.size(dlcTraj, 0)):
            replaceIdx = []
            for dim in range(3):
                relMaxs = argrelextrema(dlcVel_tmp[part, dim, :], np.greater, order = 11)[0]
                relMins = argrelextrema(dlcVel_tmp[part, dim, :], np.less, order = 11)[0]
            
                #filter out extrema cause by single frame jumps near longer jumps
                for mx in relMaxs:
                    if np.nanmean(dlcVel_tmp[part, dim, mx-5:mx+6]) < 0.75 * dlcVel_tmp[part, dim, mx]:
                        relMaxs = relMaxs[relMaxs != mx]
                for mn in relMins:
                    if np.nanmean(dlcVel_tmp[part, dim, mn-5:mn+6]) > 0.75 * dlcVel_tmp[part, dim, mn]:
                        relMins = relMins[relMins != mn]
                
                # find jumps where there is a sign change between succesive extrema and they are within 100 frames of each other
                peakIdxs = np.union1d(relMaxs, relMins)
                for firstPeak, secondPeak in zip(peakIdxs[:-1], peakIdxs[1:]):
                    if np.sign(dlcVel_tmp[part, dim, firstPeak]) != np.sign(dlcVel_tmp[part, dim, secondPeak]) and secondPeak - firstPeak < 100:
                        peak2peak = secondPeak - firstPeak
                        jumpWin = slice(int(firstPeak - np.ceil(peak2peak/2)), int(secondPeak + np.ceil(peak2peak/2)))
    
                        jumpAccel = np.divide(np.diff(dlcVel_tmp[part, dim, jumpWin]), np.diff(dlcTime[:-1][jumpWin]))
                        eventAccel = np.divide(np.diff(dlcVel_tmp[part, dim, :]), np.diff(dlcTime[:-1]))
                        eventAccel = np.delete(eventAccel, jumpWin)                        
                        
                        # compareTraj = unconnectedDLC[fNum]
                        compareTraj = unsmoothedDLC[fNum]
                        prevSamples = 10
                        nanChunk = np.where(np.isnan(compareTraj[part, dim, firstPeak - prevSamples : secondPeak]))[0]
                                
                        firstBriefChunk =  np.diff(compareTraj[part, dim, firstPeak  - prevSamples : firstPeak + 1 ]) * DLC_params.fps
                        secondBriefChunk = np.diff(compareTraj[part, dim, secondPeak - prevSamples : secondPeak + 1]) * DLC_params.fps
                        firstChunk_adjacentSum = firstBriefChunk[:-1] + firstBriefChunk[1:]
                        secondChunk_adjacentSum = secondBriefChunk[:-1] + secondBriefChunk[1:]
                        firstPeakJumpCheck =  np.any(abs(firstChunk_adjacentSum [~np.isnan(firstChunk_adjacentSum )]) > abs(dlcVel_tmp[part, dim, firstPeak ]) * 10)
                        secondPeakJumpCheck = np.any(abs(secondChunk_adjacentSum[~np.isnan(secondChunk_adjacentSum)]) > abs(dlcVel_tmp[part, dim, secondPeak]) * 10)
                        
                        varCheck = np.nanstd(abs(jumpAccel)) > 5 * np.nanstd(abs(eventAccel))
                        meanCheck = np.nanmean(abs(jumpAccel)) > np.nanmean(abs(eventAccel)) + 3.5*np.nanstd(abs(eventAccel))
                        if varCheck or meanCheck:
                            if firstPeakJumpCheck or secondPeakJumpCheck:
                                jumpWin = [i for i in range(jumpWin.start, jumpWin.stop) if ~np.isnan(dlcTraj[part, dim, i])]
                                jumpWin = slice(jumpWin[0], jumpWin[-1])
                                replaceIdx.append(jumpWin)
                                print((fNum, part, dim, firstPeak/200*1000, secondPeak/200*1000, 
                                        varCheck, meanCheck, firstPeakJumpCheck, secondPeakJumpCheck))
                                
            for idx in replaceIdx:                    
                if idx.stop - idx.start < 200:
                    for dim in range(3):
                        dlcTraj[part, dim, idx] = np.linspace(dlcTraj[part, dim, idx.start], dlcTraj[part, dim, idx.stop - 1], idx.stop - idx.start)                    
                        remSlice = np.where(~np.isnan(dlcTraj[part, dim, :]))
                        # meanSubtract[part, dim] = meanSubtract[part, dim] + np.mean(dlcTraj[part, dim, remSlice])
                        if part == 0 and dim == 1:
                            handCutoff_forPlot[fNum] = handCutoff_forPlot[fNum] - np.mean(dlcTraj[part, dim, remSlice])
                        dlcTraj[part, dim, remSlice] = dlcTraj[part, dim, remSlice] - np.repeat(np.mean(dlcTraj[part, dim, remSlice]), len(remSlice))

                        dlcVel_tmp[part, dim, idx] = np.linspace(dlcVel_tmp[part, dim, idx.start], dlcVel_tmp[part, dim, idx.stop - 1], idx.stop - idx.start)        
                        savgolFiltWin = slice(int(idx.start - DLC_params.winSize/2), int(idx.stop + DLC_params.winSize/2))
                        savgolFiltWin = [i for i in range(savgolFiltWin.start, savgolFiltWin.stop) if ~np.isnan(dlcVel_tmp[part, dim, i])]
                        dlcVel_tmp[part, dim, savgolFiltWin] = savgol_filter(dlcVel_tmp[part, dim, savgolFiltWin], DLC_params.winSize, DLC_params.polyOrder)
                else: 
                    dlcTraj[part, :, idx] = np.nan
                    dlcVel_tmp[part, :, idx] = np.nan
                    for dim in range(3):
                        remSlice = np.where(~np.isnan(dlcTraj[part, dim, :]))
                        # meanSubtract[part, dim] = meanSubtract[part, dim] + np.mean(dlcTraj[part, dim, remSlice])
                        if part == 0 and dim == 1:
                            handCutoff_forPlot[fNum] = handCutoff_forPlot[fNum] - np.mean(dlcTraj[part, dim, remSlice])
                        unprocessedDLC[fNum][part, dim, :] = unprocessedDLC[fNum][part, dim, :] - np.repeat(np.mean(dlcTraj[part, dim, remSlice]), len(unprocessedDLC[fNum][part, dim, :]))
                        dlcTraj[part, dim, remSlice] = dlcTraj[part, dim, remSlice] - np.repeat(np.mean(dlcTraj[part, dim, remSlice]), len(remSlice))
            
            goodDLCPoints = np.where(~np.isnan(dlcTraj[part, 0, :]))[0]
            dlcMatchingPoints = [idx for idx in goodDLCPoints if idx in goodXrommPoints]
            percentageTracked[fNum, part] = len(dlcMatchingPoints) / len(goodXrommPoints)
            dlcMatchingLength[fNum, part] = len(dlcMatchingPoints)
            xrommTrackingLength[fNum, part] = len(goodXrommPoints)
            dlcTrajLength[fNum, part] = len(goodDLCPoints)
            
        dlcVel.append(dlcVel_tmp)
        xrommVel.append(xrommVel_tmp)
        dlc[fNum] = dlcTraj
        xromm[fNum] = xrommTraj
        xromm_allData.append(allDataPointsXromm)
        xrommVel_allData.append(allDataPointsXrommVel)
        
        # meanPositionSubtracted.append(meanSubtract)

class trajData:
    dlc = dlc
    dlcVel = dlcVel
    unconnectedDLC = unconnectedDLC
    unsmoothedDLC  = unsmoothedDLC
    unprocessedDLC = unprocessedDLC
    
    xromm = xromm
    xrommVel = xrommVel
    xromm_allData = xromm_allData
    xrommVel_allData = xrommVel_allData
    
    handCutoff_forPlot = handCutoff_forPlot
    
    maxPosRange = []
    maxVelRange = []
        
#%% Calculate RMSE for every part and reach

print('\n Calculating DLC error relative to XROMM')

class posErrResults:
    rmse = []
    rmse_trajByPart = np.empty_like(percentageTracked) 
    rmse_avgOverTraj = []
    rmse_avgOverPart = []
    descriptiveStats = pd.DataFrame(np.empty((7, 3)), 
                                    index=['all', 'byHand', 'byNetwork', 'Pat', 'Tony', 'PatHand', 'PatNetwork'], 
                                    columns=['pos_MeanErr', 'pos_std', 'normErr'])
    
class velErrResults: 
    rmse = []
    rmse_trajByPart = np.empty_like(percentageTracked)
    rmse_avgOverTraj = []
    rmse_avgOverPart = []
    descriptiveStats = pd.DataFrame(np.empty((7, 3)), 
                                    index=['all', 'byHand', 'byNetwork', 'Pat', 'Tony', 'PatHand', 'PatNetwork'], 
                                    columns=['vel_MeanErr', 'vel_std', 'normErr'])

class trackingQuality:
    percentTracked = percentageTracked
    avgOverPart = []
    avgOverTraj = []
    
posRange = np.empty_like(percentageTracked)
velRange = np.empty_like(percentageTracked)
for trajNum in range(len(dlc)):
    dlcTraj = dlc[trajNum]
    xrommTraj = xromm[trajNum]
    dlcTrajVel = dlcVel[trajNum]
    xrommTrajVel = xrommVel[trajNum]
    
    # rmse_tmp = np.empty(np.shape(dlcTraj[:, :, 0]))
    rmse_tmp = np.empty(np.shape(dlcTraj)[0])
    rmseNorm_tmp = np.empty_like(rmse_tmp)
    # vel_rmse_tmp = np.empty(np.shape(dlcTrajVel[:, :, 0]))
    vel_rmse_tmp = np.empty_like(rmse_tmp)
    vel_rmseNorm_tmp = np.empty_like(rmse_tmp)

    print((fNum, trackedXrommPoints[trajNum].min(), trackedXrommPoints[trajNum].max()), len(xrommTraj[0, 0, :]))
    for part in range(np.shape(dlcTraj)[0]):
        
        if trajNum == 6:
            catch = []
        errorByDim = xrommTraj[part, :, :] - dlcTraj[part, :, :]
        velErrorByDim = xrommTrajVel[part, :, :] - dlcTrajVel[part, :, :]
                
        # error = errorByDim
        error = np.sqrt(np.square(errorByDim[0, :]) + np.square(errorByDim[1, :]) + np.square(errorByDim[2, :]))
        velError = np.sqrt(np.square(velErrorByDim[0, :]) + np.square(velErrorByDim[1, :]) + np.square(velErrorByDim[2, :]))
        if np.any(~np.isnan(error)):
            rmse_tmp[part] = np.nanmean(error)
            tmp_posRange = np.nanmax(xrommTraj[part, :, trackedXrommPoints[trajNum]], axis = 0) - np.nanmin(xrommTraj[part, :, trackedXrommPoints[trajNum]], axis = 0)
            tmp_velRange = np.nanmax(xrommTrajVel[part, :, trackedXrommPoints[trajNum]], axis = 0) - np.nanmin(xrommTrajVel[part, :, trackedXrommPoints[trajNum]], axis = 0)
            posRange[trajNum, part] = np.sqrt(tmp_posRange @ tmp_posRange)
            velRange[trajNum, part] = np.sqrt(tmp_velRange @ tmp_velRange)
            rmseNorm_tmp[part] = np.divide(rmse_tmp[part], posRange[trajNum, part])        
            vel_rmse_tmp[part] = np.nanmean(velError)
            vel_rmseNorm_tmp[part] = np.divide(vel_rmse_tmp[part], velRange[trajNum, part])

            # rmse_tmp[part, :] = np.sqrt(np.nanmean(np.square(error), axis = 1)).transpose()
            # rmseNorm_tmp[part, :] = np.divide(rmse_tmp[part, :], tmp_dispRange)        
            # vel_rmse_tmp[part, :] = np.sqrt(np.nanmean(np.square(velError), axis = 1)).transpose()
            # vel_rmseNorm_tmp[part, :] = np.divide(vel_rmse_tmp[part, :], tmp_velRange)


        else:
            rmse_tmp[part]         = np.nan
            rmseNorm_tmp[part]     = np.nan
            vel_rmse_tmp[part]     = np.nan
            vel_rmseNorm_tmp[part] = np.nan
            posRange[trajNum, part] = np.nan
            velRange[trajNum, part] = np.nan  
            
            # rmse_tmp[part, :]         = np.array([np.nan, np.nan, np.nan])
            # rmseNorm_tmp[part, :]     = np.array([np.nan, np.nan, np.nan])
            # vel_rmse_tmp[part, :]     = np.array([np.nan, np.nan, np.nan])
            # vel_rmseNorm_tmp[part, :] = np.array([np.nan, np.nan, np.nan])
            
    posErrResults.rmse.append(rmse_tmp)
    # posErrResults.rmse_trajByPart[trajNum, :] = np.nanmean(rmse_tmp, 1)
    posErrResults.rmse_trajByPart[trajNum, :] = rmse_tmp
    
    velErrResults.rmse.append(vel_rmse_tmp)
    # velErrResults.rmse_trajByPart[trajNum, :] = np.nanmean(vel_rmse_tmp, 1)
    velErrResults.rmse_trajByPart[trajNum, :] = vel_rmse_tmp 
        
trajData.maxPosRange = np.nanmax(posRange, axis = 0)
trajData.maxVelRange = np.nanmax(velRange, axis = 0)
        
# trackingQuality.avgOverTraj = np.divide(np.sum(np.multiply(percentageTracked, xrommTrackingLength), 1), np.sum(xrommTrackingLength, 1))
# trackingQuality.avgOverPart = np.divide(np.sum(np.multiply(percentageTracked, xrommTrackingLength), 0), np.sum(xrommTrackingLength, 0))
trackingQuality.avgOverTraj = np.divide(np.sum(np.multiply(percentageTracked, dlcMatchingLength), 1), np.sum(dlcMatchingLength, 1))
trackingQuality.avgOverPart = np.divide(np.sum(np.multiply(percentageTracked, dlcMatchingLength), 0), np.sum(dlcMatchingLength, 0))

posErrResults.rmse_avgOverTraj = np.divide(np.nansum(np.multiply(posErrResults.rmse_trajByPart, dlcTrajLength), 1), np.nansum(dlcTrajLength, 1))
posErrResults.rmse_avgOverPart = np.divide(np.nansum(np.multiply(posErrResults.rmse_trajByPart, dlcTrajLength), 0), np.nansum(dlcTrajLength, 0))
velErrResults.rmse_avgOverTraj = np.divide(np.nansum(np.multiply(velErrResults.rmse_trajByPart, dlcTrajLength), 1), np.nansum(dlcTrajLength, 1))
velErrResults.rmse_avgOverPart = np.divide(np.nansum(np.multiply(velErrResults.rmse_trajByPart, dlcTrajLength), 0), np.nansum(dlcTrajLength, 0))

allIdxs = range(len(dlc))
params.patReaches = range(len(dlc) - 4)
params.tonyReaches = range(len(dlc) - 4, len(dlc))
for storeIdx, trajIdxs in enumerate([allIdxs, params.handLabeledReaches, params.networkLabeledReaches, 
                                     params.patReaches, params.tonyReaches, 
                                     np.intersect1d(params.patReaches, params.handLabeledReaches),
                                     np.intersect1d(params.patReaches, params.networkLabeledReaches)]):
    posVec  = posErrResults.rmse_trajByPart[trajIdxs, :].flatten()
    velVec  = velErrResults.rmse_trajByPart[trajIdxs, :].flatten()
    weights = dlcTrajLength[trajIdxs, :].flatten()
    
    posStats = DescrStatsW(posVec[~np.isnan(posVec)], weights = weights[~np.isnan(posVec)])
    velStats = DescrStatsW(velVec[~np.isnan(velVec)], weights = weights[~np.isnan(velVec)])

    norm_posVec = np.divide(posVec, np.repeat(trajData.maxPosRange, len(trajIdxs)))
    norm_velVec = np.divide(velVec, np.repeat(trajData.maxVelRange,  len(trajIdxs)))
    norm_posStats = DescrStatsW(norm_posVec[~np.isnan(posVec)], weights = weights[~np.isnan(posVec)])
    norm_velStats = DescrStatsW(norm_velVec[~np.isnan(velVec)], weights = weights[~np.isnan(velVec)])
    
    posErrResults.descriptiveStats.iloc[storeIdx, :] = np.array([posStats.mean, posStats.std/np.sqrt(17), norm_posStats.mean])
    velErrResults.descriptiveStats.iloc[storeIdx, :] = np.array([velStats.mean, velStats.std/np.sqrt(17), norm_velStats.mean])    

#%% Compute inter-marker distances

meanDistances = pd.DataFrame(np.empty((4, 3)), 
                             index=['hand', 'fore', 'upper', 'torso'], 
                             columns=['m1_dist', 'm2_dist', 'm3_dist'])

handDist = np.zeros((3, len(dlc)))
foreDist = np.zeros((3, len(dlc)))
upperDist = np.zeros((3, len(dlc)))
torsoDist = np.zeros((3, len(dlc)))
totalFrames = np.zeros((4, 3))
for fNum, (dlc, meanSub) in enumerate(zip(trajData.dlc, meanPositionSubtracted)):
    for idx, (firstMark, secondMark) in enumerate(zip([0, 0, 1], [1, 2, 2])):
            
        m1 = dlc[firstMark, ...].squeeze()  + np.tile(np.reshape(meanSub[firstMark, :], (3,1)), (1, dlc.shape[-1]))
        m2 = dlc[secondMark, ...].squeeze() + np.tile(np.reshape(meanSub[secondMark, :], (3,1)), (1, dlc.shape[-1]))
        realFrames = np.intersect1d(np.where(~np.isnan(m1[0, :]))[0], np.where(~np.isnan(m2[0, :]))[0])
        handDist[idx, fNum] = np.mean(np.sqrt(np.square(m1[0, realFrames] - m2[0, realFrames]) + np.square(m1[1, realFrames] - m2[1, realFrames]) + np.square(m1[2, realFrames] - m2[2, realFrames]))) * len(realFrames)
        totalFrames[0, idx] += len(realFrames)
        
        m1 = dlc[firstMark + 3, ...].squeeze()  + np.tile(np.reshape(meanSub[firstMark, :], (3,1)), (1, dlc.shape[-1]))
        m2 = dlc[secondMark + 3, ...].squeeze() + np.tile(np.reshape(meanSub[secondMark, :], (3,1)), (1, dlc.shape[-1]))
        realFrames = np.intersect1d(np.where(~np.isnan(m1[0, :]))[0], np.where(~np.isnan(m2[0, :]))[0])
        foreDist[idx, fNum] = np.mean(np.sqrt(np.square(m1[0, realFrames] - m2[0, realFrames]) + np.square(m1[1, realFrames] - m2[1, realFrames]) + np.square(m1[2, realFrames] - m2[2, realFrames]))) * len(realFrames)
        totalFrames[1, idx] += len(realFrames)
        
        m1 = dlc[firstMark + 6, ...].squeeze()  + np.tile(np.reshape(meanSub[firstMark, :], (3,1)), (1, dlc.shape[-1]))
        m2 = dlc[secondMark + 6, ...].squeeze() + np.tile(np.reshape(meanSub[secondMark, :], (3,1)), (1, dlc.shape[-1]))
        realFrames = np.intersect1d(np.where(~np.isnan(m1[0, :]))[0], np.where(~np.isnan(m2[0, :]))[0])
        upperDist[idx, fNum] = np.mean(np.sqrt(np.square(m1[0, realFrames] - m2[0, realFrames]) + np.square(m1[1, realFrames] - m2[1, realFrames]) + np.square(m1[2, realFrames] - m2[2, realFrames]))) * len(realFrames)
        totalFrames[2, idx] += len(realFrames)

        m1 = dlc[firstMark + 9, ...].squeeze()  + np.tile(np.reshape(meanSub[firstMark, :], (3,1)), (1, dlc.shape[-1]))
        m2 = dlc[secondMark + 9, ...].squeeze() + np.tile(np.reshape(meanSub[secondMark, :], (3,1)), (1, dlc.shape[-1]))
        realFrames = np.intersect1d(np.where(~np.isnan(m1[0, :]))[0], np.where(~np.isnan(m2[0, :]))[0])
        torsoDist[idx, fNum] = np.mean(np.sqrt(np.square(m1[0, realFrames] - m2[0, realFrames]) + np.square(m1[1, realFrames] - m2[1, realFrames]) + np.square(m1[2, realFrames] - m2[2, realFrames]))) * len(realFrames)
        totalFrames[3, idx] += len(realFrames)
                        
meanDistances.iloc[0, :] = np.nansum(handDist , axis=-1) / totalFrames[0, :]
meanDistances.iloc[1, :] = np.nansum(foreDist , axis=-1) / totalFrames[1, :]
meanDistances.iloc[2, :] = np.nansum(upperDist, axis=-1) / totalFrames[2, :]
meanDistances.iloc[3, :] = np.nansum(torsoDist, axis=-1) / totalFrames[3, :]


#%%

# TO DO LIST - FIX THESE PROCESSING ERRORS
# 0) Run through all data to check if some jumps were not identified

# VIDEO NOTES
    # Pat_15
        # 002 (0) - super short reach, good except for torso_upper (error in cam2)
        # 003**   - two reaches, good
        # 007     - one reach, good, small errors in various body parts but fine
        # 008     - forearm proximal jumps briefly in cam1, upperArm medial extends past time of other markers --> check code
        # 013**   - short reach at beginning, good ()
        # 015**   - medium reach, most markers good. Torso-upper should cut off sooner --> check code. Forearm doesn't track sup/pronation well
        # 018     - 2 long reaches, great
    # Pat_14
        # 001 (7)  - Tony interferes and obscures torso, but other markers are good (pfr - big jump in torso)
        # 022**    - Not well-labeled forearm and upper arm in cam2 (reach thru left armhole)
        # 023      - Not well-labeled forearm and upper arm in cam2 (reach thru left armhole)
        # 026**    - Most/all labels need fixing for most body parts (pfr - still not great)
        # 046**    - Hand markers jump to left hand in cam1
        # 048      - good
    # Tony_14
        # 013 (13) - very short reach, good. However, upperArm proximal extends longer in time than rest of markers
        # 015      - Pat interferes and obscures torso, but other markers are good
        # 021      - Torso upper and front not labeled well in cam2, rest labels are good
        # 028      - hand markers jump at ~1250ms because left hand covers right hand
     
# Video NOTES (8/22/2020 - after first round of refinement)
    # Pat_15
        # 002 (0)  ++ - DONE - super short reach                                
        # 003**    ++ - DONE - two reaches, good
        # 007      ++ - DONE - one reach, good, small errors in various body parts but fine
        # 008      ++ - DONE - forearm proximal jumps briefly in cam1, upperArm medial extends past time of other markers --> check code
        # 013**    ++ - NOT DONE - very short reach (need to check why reach is cut off too soon)
        # 015**    ++ - DONE - medium reach, most markers good. Torso-upper should cut off sooner --> check code. Forearm doesn't track sup/pronation well
        # 018      ++ - DONE - 2 long reaches, great
    # Pat_14
        # 001 (7)  ++ - DONE - torso obscured by Tony
        # 022**    ++ - NOT DONE - everything cuts off before reach
        # 023      -- - NOT DONE - everything cuts off beginning and end of reach (cam2-415, 475, 710-800)(cam1-415,475,750-800)
        # 026**    -- - NOT DONE - Hand very low prob
        # 046**    ++ - DONE
        # 048      ++ - DONE - good
    # Tony_14
        # 013 (13) ++ - DONE - very short reach, good. However, upperArm proximal extends longer in time than rest of markers
        # 015      ++ - DONE - Pat interferes and obscures torso and upperArm Distal/proximal earlier than other parts
        # 021      ++ - NOT DONE - Check torso back labeling and forearm proximal
        # 028      ++ - DONE 
        
badEvents = [9, 10] 
#%% function definitions for making figures

def plotTrajectories(trajData, trajNum, parts, colors, figSet, pos, vel, combined):
    
    labelNames = ['hand - distal', 'hand - medial', 'hand - lateral', 
                  'forearm - distal', 'forearm - medial', 'forearm - proximal', 
                  'upperArm - distal', 'upperArm - medial', 'upperArm - proximal',
                  'torso - upper', 'torso - back', 'torso - front']
    
    if parts == 'all':
        parts = range(12)
    
    dlc = trajData.dlc
    dlcVel = trajData.dlcVel
    unconnectedDLC = trajData.unconnectedDLC
    unsmoothedDLC  = trajData.unsmoothedDLC
    unprocessedDLC = trajData.unprocessedDLC
    xromm = trajData.xromm
    xrommVel = trajData.xrommVel
    xromm_allData = trajData.xromm_allData
    
    colors = [tuple(col/255) for col in colors]
    
    jumpColors = np.array([[102,194,165],
                           [252,141,98],
                           [141,160,203]])
    jumpColors = [tuple(col/255) for col in jumpColors]
    
    lw = 2 #0.5
    axLabSize = 14
    tickLabSize = 12
    figSize3d = (7, 7) # (4.5, 4.5)
    figSize = (5, 7) # (2.25,3.5)
    thickMult = 0.009 # 0.02
    textMult = 4 # 2.5

    eLabLoc = [-0.1, 0.5] # [0.5, 0.5] [-0.08, 0.5]
    tLabLoc = [-0.1, 0.5] # [0.5, 0.5]# [-0.08, 0.5]
    
    if pos:
        minFrames = []
        maxFrames = []
        for part in parts:
            tempFrames = np.where(~np.isnan(xromm_allData[trajNum][part, 0, :]))[0]
            if len(tempFrames) != 0:
                minFrames.append(np.nanmin(tempFrames))
                maxFrames.append(np.nanmax(tempFrames))    
        minFrame = np.min(minFrames)
        maxFrame = np.max(maxFrames)
    if vel:
        minFrames = []
        maxFrames = []
        for part in parts:
            tempFrames = np.where(~np.isnan(xrommVel_allData[trajNum][part, 0, :]))[0]
            if len(tempFrames) != 0:
                minFrames.append(np.nanmin(tempFrames))
                maxFrames.append(np.nanmax(tempFrames)) 
        minVelFrame = np.min(minFrames)
        maxVelFrame = np.max(maxFrames)

    plt.style.use('seaborn-white')
    sns.set_style('ticks')
    for part in parts:        
        time = np.linspace(0, np.shape(dlc[trajNum])[2] / DLC_params.fps, num = np.shape(dlc[trajNum])[2])
        x_time = np.linspace(0, np.shape(xromm_allData[trajNum])[2] / XROMM_params.fps, num = np.shape(xromm_allData[trajNum])[2])
        
        realFrames = np.where(~np.isnan(xromm_allData[trajNum][part, 0, :]))[0]
        tAdj = x_time[realFrames[0]]
            
        errOff = 10
        if pos:
            if combined:
                fig, (axPos, axErrPos, axVel, axErrVel) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [4, 1, 4, 1]}, figsize=figSize)
            else:
                fig, (axPos, axErrPos) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(1.75,3))
                
            realFrames = np.where(~np.isnan(dlc[trajNum][part, 0, :]))
            if len(realFrames[0]) == 0:
                firstSpace = 0
                secondSpace = 0
            else:
                firstSpace  = dlc[trajNum][part, 0, realFrames].squeeze().min() - dlc[trajNum][part, 1, realFrames].squeeze().max()
                secondSpace = dlc[trajNum][part, 1, realFrames].squeeze().min() - dlc[trajNum][part, 2, realFrames].squeeze().max()

            errorByDim = xromm_allData[trajNum][part, :, realFrames].squeeze() - dlc[trajNum][part, :, realFrames].squeeze() 
            error = np.sqrt(np.square(errorByDim[:, 0]) + np.square(errorByDim[:, 1]) + np.square(errorByDim[:, 2]))

            axPos.plot(time[realFrames] - tAdj, dlc[trajNum][part, 0, realFrames].squeeze() - firstSpace  + 0.5, linestyle='-', color=colors[0], linewidth=lw)
            axPos.plot(time[realFrames] - tAdj, dlc[trajNum][part, 1, realFrames].squeeze()                    , linestyle='-', color=colors[1], linewidth=lw)
            axPos.plot(time[realFrames] - tAdj, dlc[trajNum][part, 2, realFrames].squeeze() + secondSpace - 0.5, linestyle='-', color=colors[2], linewidth=lw)
            
            # plt.plot(time[realFrames] - tAdj, unprocessedDLC[trajNum][part, 0, realFrames].squeeze() + 5, linestyle='-', color=jumpColors[0])
            # plt.plot(time[realFrames] - tAdj, unprocessedDLC[trajNum][part, 1, realFrames].squeeze()    , linestyle='-', color=jumpColors[1])
            # plt.plot(time[realFrames] - tAdj, unprocessedDLC[trajNum][part, 2, realFrames].squeeze() - 5, linestyle='-', color=jumpColors[2])          
        
            realFramesX = np.where(~np.isnan(xromm_allData[trajNum][part, 0, :]))[0]
            axPos.plot(x_time[realFramesX] - tAdj, xromm_allData[trajNum][part, 0, realFramesX].squeeze() - firstSpace  + 0.5, linestyle='-.', color=colors[0], linewidth=lw*1.25)
            axPos.plot(x_time[realFramesX] - tAdj, xromm_allData[trajNum][part, 1, realFramesX].squeeze()                    , linestyle='-.', color=colors[1], linewidth=lw*1.25)
            axPos.plot(x_time[realFramesX] - tAdj, xromm_allData[trajNum][part, 2, realFramesX].squeeze() + secondSpace - 0.5, linestyle='-.', color=colors[2], linewidth=lw*1.25)    

            axErrPos.plot(time[realFramesX][errOff:] - tAdj + np.diff(time[realFramesX[[0, errOff]]]), np.repeat(np.mean(error), len(time[realFramesX][errOff:])), linestyle='-', color=(.7, .7, .7), linewidth=lw+1) 
            axErrPos.plot(time[realFrames]  - tAdj, error, linestyle='-', color='k', linewidth=lw)
            
            # y_yStart = xromm_allData[trajNum][part, 1, realFramesX].squeeze()[0]
            y_yStart = dlc[trajNum][part, 1, realFrames].squeeze()[0]
            unit = 'cm'
            yLen = 5
            
            plot_ymax = dlc[trajNum][part, 0, realFrames].squeeze().max() - firstSpace  + 0.5
            plot_ymin = dlc[trajNum][part, 2, realFrames].squeeze().min() + secondSpace - 0.5
            
            # realFrames = np.where(~np.isnan(xromm[trajNum][part, 0, :]))
            # plt.plot(x_time[realFrames] - tAdj, xromm[trajNum][part, 0, realFrames].squeeze() + 5, '-.b')
            # plt.plot(x_time[realFrames] - tAdj, xromm[trajNum][part, 1, realFrames].squeeze(), '-.g')
            # plt.plot(x_time[realFrames] - tAdj, xromm[trajNum][part, 2, realFrames].squeeze() - 5, '-.k')
            
            # if part == 0:
            #     axPos.plot(x_time[realFrames], np.repeat(handCutoff_forPlot[trajNum], len(x_time[realFrames])), '-r')
            
            axPos.set_yticklabels([])
            axPos.set_xticklabels([])
            if combined:
                axErrPos.set_xticklabels([])
            else:
                axErrPos.set_xlabel('Time (s)')


            axErrPos.set_xticks(np.arange(time[minFrame] - tAdj, time[maxFrame] - tAdj, 1))                
            axErrPos.set_yticks([round(np.mean(error), 1), 2])
            axErrPos.tick_params(bottom=True, left=True, length = 2, width=1, direction='out')
            if figSet == 0:
                axErrPos.set_ylabel('Error (cm)', fontsize = axLabSize)
                axErrPos.yaxis.set_label_coords(eLabLoc[0], eLabLoc[1])
                axPos.set_ylabel('Position')
                axPos.yaxis.set_label_coords(tLabLoc[0], tLabLoc[1])
            for item in [axPos.xaxis, axPos.yaxis]:
                item.label.set_fontsize(axLabSize)
            for label in (axErrPos.get_xticklabels() + axErrPos.get_yticklabels()):
                label.set_fontsize(tickLabSize)
            
            axPos.spines['bottom'].set_linewidth(1)
            axErrPos.spines['bottom'].set_linewidth(1)
            axErrPos.spines['left'].set_linewidth(1)
            
            sns.despine(bottom=True, left=True, offset={'bottom': 0}, ax=axPos)
            sns.despine(bottom=False, left=False, ax=axErrPos)

            print(('position (min,max,y_yStart)', plot_ymin, plot_ymax, y_yStart))
    
            y_xStart = time[minFrame] - tAdj  
            y_yStart = -1.58
            yThick   = thickMult *  (time[maxFrame] - time[minFrame]) #0.025
                
            yScale = matplotlib.patches.Rectangle((y_xStart, y_yStart), yThick, yLen, angle=0.0, clip_on=False, facecolor=(0.1, 0.1, 0.1))
            axPos.add_patch(yScale)
            axPos.text(y_xStart - textMult*yThick, y_yStart + 0.3*yLen, str(yLen) + ' ' + unit, clip_on=False, rotation='vertical', fontsize=tickLabSize)    
            
            axPos.set_xlim(   time[minFrame] - tAdj, time[maxFrame] - tAdj) 
            axErrPos.set_xlim(time[minFrame] - tAdj, time[maxFrame] - tAdj)
            
            axPos.set_ylim(-9, 7)
                 
            # plt.tight_layout()
            # plt.show()

        if vel:
            if not combined:
                fig, (axVel, axErrVel) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(4,5))
                
            realFrames = np.where(~np.isnan(dlcVel[trajNum][part, 0, :]))
            firstSpace  = dlcVel[trajNum][part, 0, realFrames].squeeze().min() - dlcVel[trajNum][part, 1, realFrames].squeeze().max()
            secondSpace = dlcVel[trajNum][part, 1, realFrames].squeeze().min() - dlcVel[trajNum][part, 2, realFrames].squeeze().max()

            errorByDim = xrommVel[trajNum][part, :, realFrames].squeeze() - dlcVel[trajNum][part, :, realFrames].squeeze()
            error = np.sqrt(np.square(errorByDim[:, 0]) + np.square(errorByDim[:, 1]) + np.square(errorByDim[:, 2]))

            axVel.plot(time[realFrames] - tAdj, dlcVel[trajNum][part, 0, realFrames].squeeze() - firstSpace  + 0.5, linestyle='-', color=colors[0], linewidth=lw)
            axVel.plot(time[realFrames] - tAdj, dlcVel[trajNum][part, 1, realFrames].squeeze()                    , linestyle='-', color=colors[1], linewidth=lw)
            axVel.plot(time[realFrames] - tAdj, dlcVel[trajNum][part, 2, realFrames].squeeze() + secondSpace - 0.5, linestyle='-', color=colors[2], linewidth=lw)
            
            realFramesX = np.where(~np.isnan(xrommVel_allData[trajNum][part, 0, :]))[0]
            axVel.plot(x_time[realFramesX] - tAdj, xrommVel_allData[trajNum][part, 0, realFramesX].squeeze() - firstSpace  + 0.5, linestyle='-.', color=colors[0], linewidth=lw*1.25)
            axVel.plot(x_time[realFramesX] - tAdj, xrommVel_allData[trajNum][part, 1, realFramesX].squeeze()                    , linestyle='-.', color=colors[1], linewidth=lw*1.25)
            axVel.plot(x_time[realFramesX] - tAdj, xrommVel_allData[trajNum][part, 2, realFramesX].squeeze() + secondSpace - 0.5, linestyle='-.', color=colors[2], linewidth=lw*1.25)

            axErrVel.plot(time[realFramesX][errOff:] - tAdj + np.diff(time[realFramesX[[0, errOff]]]), np.repeat(np.mean(error), len(time[realFramesX][errOff:])), linestyle='-', color=(.7, .7, .7), linewidth=lw+1) 
            axErrVel.plot(time[realFrames] - tAdj, error, linestyle='-', color='k', linewidth=lw)
            
            y_yStart = dlcVel[trajNum][part, 1, realFrames].squeeze()[0]
            unit = 'cm/s'
            yLen = 30
            plot_ymax = dlcVel[trajNum][part, 0, realFrames].squeeze().max() - firstSpace  + 0.5
            plot_ymin = dlcVel[trajNum][part, 2, realFrames].squeeze().min() + secondSpace - 0.5

            axVel.set_yticklabels([])
            axVel.set_xticklabels([])
            axErrVel.set_xticks(np.arange(time[minVelFrame] - tAdj, time[maxVelFrame] - tAdj, 1))
            axErrVel.set_yticks([round(np.mean(error), 1), int(15)])
            axErrVel.tick_params(bottom=True, left=True, length = 2, width=1, direction='out')
            axErrVel.set_xlabel('Time (s)', fontsize = axLabSize)
            if figSet == 0:
                axErrVel.set_ylabel('Error (cm/s)', fontsize = axLabSize)
                axErrVel.yaxis.set_label_coords(eLabLoc[0], eLabLoc[1])
                axVel.set_ylabel('Velocity', fontsize = axLabSize)
                axVel.yaxis.set_label_coords(tLabLoc[0], tLabLoc[1])
            for item in [axVel.xaxis, axVel.yaxis]:
                item.label.set_fontsize(axLabSize)
            for label in (axErrVel.get_xticklabels() + axErrVel.get_yticklabels()):
                label.set_fontsize(tickLabSize)
            
            axVel.spines['bottom'].set_linewidth(1)
            axErrVel.spines['bottom'].set_linewidth(1)
            axErrVel.spines['left'].set_linewidth(1)
            
            sns.despine(bottom=True, left=True, offset={'bottom': 1}, ax=axVel)
            sns.despine(bottom=False, left=False, ax=axErrVel)

            print(('velocity (min,max,y_yStart)', plot_ymin, plot_ymax, y_yStart))    
    
            y_xStart = time[minVelFrame] - tAdj  
            y_yStart = 2 #7.49
            yThick   = thickMult *  (time[maxVelFrame] - time[minVelFrame]) #0.025
                
            yScale = matplotlib.patches.Rectangle((y_xStart, y_yStart), yThick, yLen, angle=0.0, clip_on=False, facecolor=(0.1, 0.1, 0.1))
            axVel.add_patch(yScale)
            axVel.text(y_xStart - textMult*yThick, y_yStart + 0.1*yLen, str(yLen) + ' ' + unit, clip_on=False, rotation='vertical', fontsize=tickLabSize)    
            
            axVel.set_xlim(   time[minVelFrame] - tAdj, time[maxVelFrame] - tAdj)        
            axErrVel.set_xlim(time[minVelFrame] - tAdj, time[maxVelFrame] - tAdj)        

            axVel.set_ylim(-59.2, 62.2)
            # plt.tight_layout()

            # plt.show()
        plt.tight_layout()
        plt.show()
        fig.savefig(params.saveFigPath + 'trajNum_' + str(trajNum) + '_part_' + str(part) + '_xyz.png', bbox_inches='tight', dpi=200)

    
    ################
    
    # part = 1
    # dim = 2
    # realFrames = np.where(~np.isnan(dlc[trajNum][part, dim, :]))[0]
    # realFrames = realFrames[realFrames < unconnectedDLC[trajNum].shape[-1]]
    # plt.plot(time[realFrames], unconnectedDLC[trajNum][part, dim, realFrames].squeeze())
    # plt.show()
    
    # realFrames = np.where(~np.isnan(dlcVel[trajNum][part, dim, :]))
    # plt.plot(time[realFrames], dlcVel[trajNum][part, dim, realFrames].squeeze())
    # plt.show()
    
    ################
    
    part = parts[0]
    
    # if not combined:
    fig = plt.figure(figsize=figSize3d)
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.view_init(12, -2)
    # else:
    #     ax3d.remove()
    #     ax3d = fig.add_subplot(511, projection='3d', size = (4, 5))
    
    # idx = [int(0 * 1e-3 * DLC_params.fps), int(4500 * 1e-3 * DLC_params.fps)]
    idx = [0, len(dlc[trajNum][part, 0, :])]
    
    blue = tuple(np.array([0, 0, 139])/255)
    lightblue = tuple(np.array([0,191,255])/255)
    
    firstReal = np.where(~np.isnan(dlc[trajNum][part, 0, :]))[0].min()
    
    minAx = np.min([np.nanmin(dlc[trajNum][part, 0, idx[0]:idx[1]]), np.nanmin(dlc[trajNum][part, 1, idx[0]:idx[1]]), np.nanmin(dlc[trajNum][part, 2, idx[0]:idx[1]])])
    maxAx = np.max([np.nanmax(dlc[trajNum][part, 0, idx[0]:idx[1]]), np.nanmax(dlc[trajNum][part, 1, idx[0]:idx[1]]), np.nanmax(dlc[trajNum][part, 2, idx[0]:idx[1]])])
    ax3d.plot(dlc[trajNum][part, 0, idx[0]:idx[1]], dlc[trajNum][part, 1, idx[0]:idx[1]], dlc[trajNum][part, 2, idx[0]:idx[1]], color = blue, linestyle = '-', linewidth=2)
    ax3d.plot(xromm[trajNum][part, 0, idx[0]:idx[1]], xromm[trajNum][part, 1, idx[0]:idx[1]], xromm[trajNum][part, 2, idx[0]:idx[1]], color = lightblue, linestyle = '-.', linewidth=1.5)
    ax3d.plot(dlc[trajNum][part, 0, firstReal:firstReal+1], dlc[trajNum][part, 1, firstReal:firstReal+1], dlc[trajNum][part, 2, firstReal:firstReal+1], color = blue, marker = 'o', markersize=12, linestyle = '-', linewidth=2)
    
    ax3d.set_xlim(minAx - 0.1*minAx, maxAx + 0.1*maxAx)
    ax3d.set_ylim(minAx - 0.1*minAx, maxAx + 0.1*maxAx)
    ax3d.set_zlim(minAx - 0.1*minAx, maxAx + 0.1*maxAx)
    ax3d.set_xlabel('x (cm)')
    ax3d.set_ylabel('y (cm)')
    ax3d.set_zlabel('z (cm)')    
    for item in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        item.label.set_fontsize(axLabSize)
    for label in (ax3d.get_xticklabels() + ax3d.get_yticklabels() + ax3d.get_zticklabels()):
        label.set_fontsize(tickLabSize)
    ax3d.legend(['DLC', 'XROMM'], loc='upper right', bbox_to_anchor=(0.85, 0.76), fontsize = 14, shadow=False)
    plt.show()
    fig.savefig(params.saveFigPath + 'trajNum_' + str(trajNum) + '_part_' + str(parts[0]) + '_3D.png')

#%% Plot trajectories


### IMPORTANT - need to edit the linewidth and text size, figsize etc for Nicho's figures. 
# Edit so that I can change easily between ppt size and paper size

dirColors = np.array([[27 , 158, 119],
                      [217, 95 , 2  ],
                      [117, 112, 179]])

# 12, 3 for bad example
# (2, 5, or 6), 0 for good example
trajNum = 11
part = 'all' # can be a list of integers, or 'all'
figSet = 0
print('\n Plotting')
plotTrajectories(trajData, trajNum, part, dirColors, figSet, pos=True, vel=True, combined=True)

#%%

def labelExample_with_basisAxes(imgPath, axes, dots, colors, savePath, imNum):
    img = cv2.imread(imgPath)
    
    colors = colors[:, (2, 1, 0)].astype(np.int64)  

    if imNum in [0, 1]:
        dotSize = 15
    elif imNum in [2, 3]:
        dotSize = 11
        

    for endPts, dot, color in zip(axes[::-1], dots, colors[::-1]):
        
        color = tuple([int(x) for x in color])
        img = cv2.circle(img, tuple(dot), dotSize, tuple([int(x) for x in [0,0,0]]), thickness=-1)        
        img = cv2.arrowedLine(img, tuple(endPts[:2]), tuple(endPts[2:]), color=color, thickness=8)
    
    if imNum == 3:
        color = tuple([int(x) for x in colors[1, :]])
        img = cv2.arrowedLine(img, tuple(axes[1, :2]), tuple(axes[1, 2:]), color=color, thickness=8)
        
    img = cv2.circle(img, tuple(dots[0,:]), dotSize, tuple([int(x) for x in [0,0,0]]), thickness=-1)        
    
    plotImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(plotImg)
    plt.show()
    
    cv2.imwrite(savePath, img)
    
def waterfallPlot(category, colors, reorder, part):

    lw = 2 #0.5
    axLabSize = 14
    tickLabSize = 12
    figSize = (3, 3*2.5) 
    yThick = 0.07
    textMult = 4
     

    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(111)
        
    if np.any(colors > 1):
        colors = colors / 255
    
    dlc_reorder = [trajData.dlc[idx] for idx in reorder]
    
    prevMin = 0
    xMax = 0
    traj_offset = []
    for dlc in dlc_reorder:
        realFrames = np.where(~np.isnan(dlc[part, 1, :]))[0]
        time = np.linspace(0, len(realFrames) / DLC_params.fps, num = len(realFrames))
        traj = dlc[part, 1, realFrames].squeeze()
        highPoint = traj.max()
        traj = traj + prevMin - highPoint - 1
        prevMin = traj.min()        
        traj_offset.append(traj) 
        if time[-1] > xMax:
            xMax = time[-1]
    
    for tNum, (traj, cat) in enumerate(zip(traj_offset, category)):
        time = np.linspace(0, len(traj) / DLC_params.fps, num = len(traj))
        ax.plot(time, traj - prevMin, color = colors[cat], linestyle = '-', linewidth=lw)
        if tNum == 6:
            y_yStart = traj[0] - prevMin    
    
    yMax = -prevMin
    yMin = 0
    
    # realFrames = np.where(~np.isnan(trajData.dlc[0][part, 1, :]))[0]
    # traj = trajData.dlc[0][part, 1, realFrames].squeeze()
    # traj_offset = [traj - traj.max()]

    # for tNum, (dlc, prev_traj) in enumerate(zip(trajData.dlc[1:], traj_offset)):
    #     realFrames = np.where(~np.isnan(dlc[part, 1, :]))[0]
    #     time = np.linspace(0, len(realFrames) / DLC_params.fps, num = len(realFrames))
    #     traj = dlc[part, 1, realFrames].squeeze()
    #     if len(traj) < len(prev_traj):
    #         traj = np.append(traj, np.repeat(np.nan, abs(len(traj) - len(prev_traj))))
    #     else:
    #         prev_traj = np.append(prev_traj, np.repeat(np.nan, abs(len(traj) - len(prev_traj))))
    #     spacing = np.nanmax(traj - prev_traj)
    #     traj_offset.append(traj - spacing - 1)
        
    #     if tNum+1 == len(trajData.dlc) - 1:
    #         lowPoint = np.nanmin(traj_offset[-1]) 
               
    # xMax = 0
    # for traj, cat in zip(traj_offset, category):
    #     time = np.linspace(0, len(traj) / DLC_params.fps, num = len(traj))
    #     ax.plot(time, traj - lowPoint, color = colors[cat], linestyle = '-', linewidth=2)
    #     if time[-1] > xMax:
    #         xMax = time[-1]
    
    # yMax = -lowPoint
    # yMin = 0
    
            
    ax.set_xlim(0, xMax)
    ax.set_ylim(yMin - 1, yMax + 1)
    ax.set_xlabel('Time (s)')
    # ax.set_ylabel('y (cm)')
    ax.set_yticklabels([])
    # ax.set_xticklabels([])
    ax.set_xticks(np.arange(0, xMax, 1))
    ax.tick_params(bottom=True, left=False, width=2)
    for item in [ax.xaxis, ax.yaxis]:
        item.label.set_fontsize(axLabSize)
    for label in (ax.get_xticklabels()):
        label.set_fontsize(tickLabSize)
    
    ax.spines['bottom'].set_linewidth(2)
    
    sns.despine(bottom=False, left=True, offset={'bottom': 5})

    x_xStart =  1 
    x_yStart = -6
    y_xStart = -0.25  
    # y_yStart = 50
    xLen     = 1
    xThick   = 0.5
    yLen     = 9

    # xScale = matplotlib.patches.Rectangle((x_xStart, x_yStart), xLen, xThick, angle=0.0, clip_on=False, facecolor='black')
    # ax.add_patch(xScale)
    # plt.text(x_xStart + 0.2*xLen, x_yStart - 3*xThick, str(xLen) + ' sec', clip_on=False)

    yScale = matplotlib.patches.Rectangle((y_xStart, y_yStart), yThick, yLen, angle=0.0, clip_on=False, facecolor='black')
    ax.add_patch(yScale)
    plt.text(y_xStart - textMult*yThick, y_yStart + 0.25*yLen, str(yLen) + ' cm', clip_on=False, rotation='vertical', fontsize=tickLabSize)
    
    # ax.add_patch(plt.Rectangle((0,-15),1, 3,facecolor='silver', clip_on=False,linewidth = 0))
    
    # ax.legend(['DLC', 'XROMM'], loc='upper right', bbox_to_anchor=(0.9, 0.75), fontsize = 14, shadow=False)
    plt.show()

#%% Figure 1

imgPaths = [r'C:/Users/daltonm/Documents/Lab_Files/updated_trajectory_analysis/trajectories_code_and_figures/DLC_validation/origin_and_axes_cam1.png',
            r'C:/Users/daltonm/Documents/Lab_Files/updated_trajectory_analysis/trajectories_code_and_figures/DLC_validation/origin_and_axes_cam2.png',
            r'C:/Users/daltonm/Documents/Lab_Files/updated_trajectory_analysis/trajectories_code_and_figures/DLC_validation/xromm_origin_and_axes.png']

savePaths = [params.saveFigPath + 'DLC_origin_and_axes_cam1_final.png',
             params.saveFigPath + 'DLC_origin_and_axes_cam2_final.png',
             params.saveFigPath + 'XROMM_origin_and_axes_cam1_final.png',
             params.saveFigPath + 'XROMM_origin_and_axes_cam2_final.png']
             
vidPath = [r'C:/Users/daltonm/Documents/Lab_Files/updated_trajectory_analysis/trajectories_code_and_figures/DLC_validation/2019_04_15_session1_event003_cam1DeepCut_resnet50_validation_PatJan23shuffle1_150000_labeled.mp4',
           r'C:/Users/daltonm/Documents/Lab_Files/updated_trajectory_analysis/trajectories_code_and_figures/DLC_validation/2019_04_15_session1_event003_cam2DeepCut_resnet50_validation_PatJan23shuffle1_150000_labeled.mp4',
           r'Z:/marmosets/XROMM_and_RGB_sessions/XROMM_videos/20190415_fullRecording/2019-04-15_11-47_Evt03/2019-04-15_11-47_Evt03-Camera1.avi',
           r'Z:/marmosets/XROMM_and_RGB_sessions/XROMM_videos/20190415_fullRecording/2019-04-15_11-47_Evt03/2019-04-15_11-47_Evt03-Camera2.avi']

frame = 124
image_names = [params.saveFigPath + 'dlc_event003_cam1_frame%d.jpg'   % frame,
               params.saveFigPath + 'dlc_event003_cam2_frame%d.jpg'   % frame,
               params.saveFigPath + 'xromm_event003_cam1_frame%d.jpg' % frame,
               params.saveFigPath + 'xromm_event003_cam2_frame%d.jpg' % frame]

#%%

# for vid, name in zip(vidPath, image_names):
#     vidcap = cv2.VideoCapture(vid)
#     success,image = vidcap.read()
#     count = 0
#     while success:
#       success,image = vidcap.read()
#       print('Read a new frame: ', success)
#       if count == frame:
#           cv2.imwrite(params.saveFigPath + name, image)     # save frame as JPEG file      
#       count += 1

#%%

# camAxes = [np.empty((3, 4), dtype=np.int64), 
#            np.empty((3, 4), dtype=np.int64), 
#            np.empty((3, 4), dtype=np.int64), 
#            np.empty((3, 4), dtype=np.int64)]

# origin = np.array([[814 , 565],
#                    [388 + 449 , 569 + 54],
#                    [541 , 396],
#                    [311 , 340]])

# xEnd   = np.array([[799 , 715],
#                    [207 + 449, 555 + 54],
#                    [705 , 415],
#                    [221 , 330]])
 
# yEnd   = np.array([[1096 , 585],
#                    [450 + 449, 720 + 54],
#                    [705 , 400],
#                    [172 , 338]])

# zEnd   = np.array([[816 , 375],
#                    [425 + 449, 441 + 54],
#                    [541 , 270],
#                    [316 , 205]])

# for axes, o, x, y, z in zip(camAxes, origin, xEnd, yEnd, zEnd):
#     axes[0, :] = np.hstack((o, x))
#     axes[1, :] = np.hstack((o, y))
#     axes[2, :] = np.hstack((o, z))

# dots1  = np.array([[814 , 565],
#                    [1116 , 585],
#                    [1205, 410]]).astype(np.int64)

# dots2  = np.array([[388 + 449 , 569 + 54],  
#                    [457 + 449 , 737 + 54],
#                    [985 , 654]]).astype(np.int64)

# xdots1  = np.array([[541 , 396],
#                     [720 , 400],
#                     [756 , 218]]).astype(np.int64)

# xdots2  = np.array([[311 , 340],
#                     [162 , 338],
#                     [137 , 179]]).astype(np.int64)

# for axes, imgPath, dots, savePath, imNum in zip(camAxes, image_names, [dots1, dots2, xdots1, xdots2], savePaths, [0, 1, 2, 3]):
#     plt.figure()
#     labelExample_with_basisAxes(imgPath, axes, dots, dirColors, savePath, imNum) 

#%%

# set up following variables for the waterfall plot
category = [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0] # define each trajectory as 0 (Pat, hand), 1 (Pat, network), or 2 (Tony)
# category = [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 2, 2, 2, 2]
# offsets = np.linspace(60, 0, 17)
# offsets = np.array([offsets[i] for i in [0, 7, 1, 2, 8, 9, 3, 4, 10, 5, 11, 12, 6, 13, 14, 15, 16]])
# probably want to permute offsets to place categories together rather than in sequential order
# categoryColors = np.array([[ 31, 120, 180],
#                            [166, 206, 227],
#                            [ 51, 160,  44]]) # need three colors - two that are different shades of similar color, one different (Tony)
categoryColors = np.vstack((dirColors[1, :], [253, 172, 110]))

reorder = [0, 2, 3, 6, 7, 10, 11, 12, 13, 14, 1, 4, 5, 8, 9]
# reorder = [0, 2, 3, 6, 7, 9, 12, 1, 4, 5, 8, 10, 11, 13, 14, 15, 16]
category = [category[idx] for idx in reorder]

sns.set()
sns.set_context('paper')
sns.set_style('white')
sns.set_style('ticks')

waterfallPlot(category, categoryColors, reorder, 0)

#%%

plt.figure(figsize=(8, 7))
plt.bar([1, 2, 3, 4], 
        height = velErrResults.descriptiveStats.iloc[[0, 1, 2, 4], 0], 
        color = np.vstack(([.4,.4,.4], categoryColors/255)),
        yerr = velErrResults.descriptiveStats.iloc[[0, 1, 2, 4], 1])
plt.show()

print('\n All done!')