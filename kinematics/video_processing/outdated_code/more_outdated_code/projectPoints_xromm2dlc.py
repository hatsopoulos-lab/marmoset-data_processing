import numpy as np
import pandas as pd
import seaborn as sns
import glob
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import itertools
from random import sample
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu, kruskal
import matplotlib
import copy
import warnings
from statsmodels.stats.weightstats import DescrStatsW
import mpl_toolkits.mplot3d.axes3d as p3

downSampRate = 200    
        
# to do list
# Completed inter-marker distances
# Now need to:
    # - figure out what the basis vectors are - identify the three points at which the hand position is at 
    #        zero for a particular dimension, find origin from that and use that to help estimate axes.
    #        Generally, +x is forward, +y is down, and +z is right (toward left-hand side of apparatus)
    # - take a look at other stats necessary

# Notes on best image set, no basis rotation. 
# 04_15 - image1 = .345 with no fix
# 04_14 - image2 = .360 with flip (0,-2,1)
       
class params:
    x_traj_path = r'Z:/marmosets/XROMM_and_RGB_sessions/XROMM_videos/validation_trajectories/' 
    labelOrder = [12, 11, 10, 9, 8, 7, 6, 5, 3, 4, 2, 1, 0]
    marm = 'tony'
    if marm == 'pat':
        event = '7'
        labeled_frames_base =  r'Z:/marmosets/deeplabcut_results/validation_Pat-Dalton-2020-01-23/labeled-data/2019_04_15_session1_event' + event.zfill(3)
        rvecs = np.array([[0.5067127843900923, -0.3902233688702131, -0.12323894076406157],
                          [0.3693965629457933, 0.8748058101230456, 0.3956494410660027]], dtype=np.float32)
        tvecs = np.array([[-4.300494331431178, 1.4003013170662144, 34.64523522268213],
                          [-7.1677885016108265, -5.456703104195415, 33.9316818035927]], dtype=np.float32)    
        camMats = np.array([[ [ 2408.31086, 0.0, 744.606275], [0.0, 2410.48885, 521.416025], [ 0.0, 0.0, 1.0]], 
                            [ [ 1618.88246, 0.0, 787.789688], [0.0, 1604.08944, 556.236720], [0.0, 0.0, 1.0]]], dtype = np.float32)
        camDists = np.array([[ -0.295935491, -0.682596812, -0.00183053340, -0.00195633780, -2.45871901], 
                              [ -0.342793230, -0.551805670, 0.00208599118, -0.00220793974, 5.36952758]], dtype = np.float32)
    elif marm == 'tony':
        event = '21'
        labeled_frames_base = r'Z:/marmosets/deeplabcut_results/validation_Tony-Dalton-2020-01-05/labeled-data/2019_04_14_session1_event' + event.zfill(3)
        rvecs = np.array([[1.828952184318242, -0.9655738973550269, 0.5430500204720892],
                          [1.9549978901014753, 0.5182194712846203, -0.015591874583159708]], dtype=np.float32)
        tvecs = np.array([[1.5557070238429147, 9.251689275812153, 36.569331730559405],
                          [-5.67497848410687, 4.824943386986453, 35.710803979112946]], dtype=np.float32)    
        camMats = np.array([[ [ 2374.01990, 0.0, 722.472508], [0.0, 2364.61960, 608.605089], [0.0, 0.0, 1.0]], 
                            [ [ 1594.95385, 0.0, 801.129953], [0.0, 1594.23315, 580.006439], [0.0, 0.0, 1.0]]], dtype = np.float32)
        camDists = np.array([[-4.73839306e-01, 5.21597237e+00, 1.12650634e-03, 2.80757313e-03, -4.76103219e+01],
                             [-0.31741825, -0.64961686, -0.00421487, -0.00304917,  3.03281784]], dtype = np.float32)
#%% Load XROMM trajectories

print('\n Loading XROMM trajectories')

xromm_traj_file = glob.glob(params.x_traj_path + '*event_' + params.event.zfill(2) + '.csv')
tmp_traj = np.loadtxt(xromm_traj_file[0], delimiter = ',', skiprows = 1)
    
traj = np.empty((int(np.size(tmp_traj, 1) / 3), 3, np.size(tmp_traj, 0)), dtype=np.float32)
for part in range(int(np.size(tmp_traj, 1) / 3)):
    traj[part, :, :] = tmp_traj[:, 3*part : 3*part+3].transpose()
            
traj = traj[params.labelOrder, :, :]  

#%% Load dlc extracted images (no labels)

dlc_cam1 = sorted(glob.glob(params.labeled_frames_base + '_cam1/*.png'))
dlc_cam2 = sorted(glob.glob(params.labeled_frames_base + '_cam2/*.png'))

ct = 1
for path1, path2 in zip(dlc_cam1, dlc_cam2):
    print(ct)
    
    im1 = cv2.imread(path1)
    im2 = cv2.imread(path2)
    
    frameNum = int(path1.split('img')[1][:4])
    framePoints = traj[..., frameNum]
    projected1, tmp = cv2.projectPoints(framePoints, params.rvecs[0], params.tvecs[0], params.camMats[0], params.camDists[0])
    projected2, tmp = cv2.projectPoints(framePoints, params.rvecs[1], params.tvecs[1], params.camMats[1], params.camDists[1])

    for pt1, pt2 in zip(projected1, projected2):
        im1 = cv2.circle(im1, tuple(map(tuple, pt1))[0], 2, (255, 0, 0), 2)
        im2 = cv2.circle(im2, tuple(map(tuple, pt2))[0], 2, (255, 0, 0), 2)
    
    cv2.imwrite(os.path.join('Z:/dalton_moore/Publications_and_grants/dlc_validation/projectedPoints_xromm2dlc', params.marm, 'cam1', 
                             'event' + params.event.zfill(3) + '_' + 'img' + str(frameNum).zfill(4)) + '.jpg', im1)
    cv2.imwrite(os.path.join('Z:/dalton_moore/Publications_and_grants/dlc_validation/projectedPoints_xromm2dlc', params.marm, 'cam2', 
                             'event' + params.event.zfill(3) + '_' + 'img' + str(frameNum).zfill(4)) + '.jpg', im2)
    ct += 1