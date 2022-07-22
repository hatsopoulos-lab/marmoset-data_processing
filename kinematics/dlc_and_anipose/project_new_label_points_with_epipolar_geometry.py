# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:39:52 2022

@author: Dalton
"""

import cv2
import numpy as np
import os
import pickle
import glob
import pd
import re
from deeplabcut import auxiliaryfunctions, auxiliaryfunctions_3d

project_path_3D = r'Z:/marmosets/deeplabcut_computer_transfers/simple_joints_model-Dalton-2021-04-08/moths3D-Dalton-2021-04-08-3d'
project_path = r'Z:/marmosets/deeplabcut_computer_transfers/simple_joints_model-Dalton-2021-04-08'
date_to_label = '2021_02_11'
source_cams = [1, 2]
target_cams = [3, 4, 5]

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

def retrieve_calibration(project_path_3D, source_cams, target_cams):

    cfg_3d = auxiliaryfunctions.read_config(os.path.join(project_path_3D, 'config.yaml'))
    cams = cfg_3d["camera_names"]
    path_stereo_file = os.path.join(project_path_3D, 'camera_matrix', "stereo_params.pickle")
    stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)    

    camera_pairs = []
    source_in_pair = []
    fundMats = []
    for source in source_cams:
        for target in target_cams:
            if source < target:
                camera_pairs.append(cams[source-1] + "-" + cams[target-1])
                source_in_pair.append(1)
            else:
                camera_pairs.append(cams[target-1] + "-" + cams[source-1])
                source_in_pair.append(2)

            fundMats.append(stereo_file[camera_pairs[-1]]["F"])
            
    return cams, camera_pairs, source_in_pair, fundMats 

def find_epipolar_intersection(epLines_source1, epLines_source2, 
                               sourcePts1, sourcePts2, 
                               target_image_path, target_offsets):
    
    im = cv2.imread(target_image_path)
    height, width, depth = im.shape
    for line, pt in zip(epLines_source1, sourcePts1):
        if pt[0] > -1000:
            coeffs = line[0]
            x0, y0 = map(int, [0 - target_offsets[0], 
                               -coeffs[2] / coeffs[1] - target_offsets[1]]
                         )
            x1, y1 = map(int, [width, 
                               -(coeffs[2] + coeffs[0] * (width + target_offsets[0])) / coeffs[1] - target_offsets[1]]
            )
            
            epLine1_ptA = (x0, y0)
            epLine1_ptB = (x1, y1)
            
            goodSource1 = True
            
        else:
            goodSource1 = False

            
    for line, pt in zip(epLines_source2, sourcePts2):
        if pt[0] > -1000:
            coeffs = line[0]
            x0, y0 = map(int, [0 - target_offsets[0], 
                               -coeffs[2] / coeffs[1] - target_offsets[1]]
                         )
            x1, y1 = map(int, [width, 
                               -(coeffs[2] + coeffs[0] * (width + target_offsets[0])) / coeffs[1] - target_offsets[1]]
            )
            
            epLine2_ptA = (x0, y0)
            epLine2_ptB = (x1, y1)
            
            goodSource2 = True
            
        else:
            goodSource2 = False

    if goodSource1 and goodSource2:
        intersect_pt = get_intersect(epLine1_ptA, epLine1_ptB, epLine2_ptA, epLine2_ptB)
    else:
        intersect_pt = np.nan # look into whether this is the correct thing to fill in
    
    return intersect_pt

def project_points(source_nums_in_pair, 
                   fundMats, 
                   source_event_label_files,
                   target_label_file,
                   project_path):

    target_images = glob.glob(os.path.join(os.path.split(target_label_file)[0], 'img*'))
    
    cfg = auxiliaryfunctions.read_config(os.path.join(project_path, 'config.yaml'))
    
    # Get crop params for target camera
    foundEvent = 0
    eventSearch = re.compile(os.path.split(os.path.split(target_label_file)[0])[1])
    cropPattern = re.compile("[0-9]{1,4}")
    with open(cfg, "rt") as config:
        for line in config:
            if foundEvent == 1:
                crop_targetCam = np.int32(re.findall(cropPattern, line))
                break
            if eventSearch.search(line) != None:
                foundEvent = 1
    
    sourceCam_pts_list = []
    for source_file in source_event_label_files:
        try:
            source_dataframe = pd.read_hdf(source_file)
            source_dataframe.sort_index(inplace=True)
        except IOError:
            print(
                "source camera images have not yet been labeled, or you have opened this folder in the wrong mode!"
            )

        # Find offset terms for drawing epipolar Lines
        # Get crop params for camera being labeled
        foundEvent = 0
        eventSearch = re.compile(os.path.split(os.path.split(source_file)[0])[1])
        cropPattern = re.compile("[0-9]{1,4}")
        with open(cfg, "rt") as config:
            for line in config:
                if foundEvent == 1:
                    crop_sourceCam = np.int32(re.findall(cropPattern, line))
                    break
                if eventSearch.search(line) != None:
                    foundEvent = 1

        targetCam_offsets = [crop_targetCam[0], crop_targetCam[2]]
        sourceCam_offsets = [crop_sourceCam[0], crop_sourceCam[2]]
    
        sourceCam_pts = np.asarray(source_dataframe, dtype=np.int32)
        sourceCam_pts = sourceCam_pts.reshape(
            (sourceCam_pts.shape[0], int(sourceCam_pts.shape[1] / 2), 2)
        )
        sourceCam_pts = np.moveaxis(sourceCam_pts, [0, 1, 2], [1, 0, 2])
        sourceCam_pts[..., 0] = sourceCam_pts[..., 0] + sourceCam_offsets[0]
        sourceCam_pts[..., 1] = sourceCam_pts[..., 1] + sourceCam_offsets[1]
    
        sourceCam_pts_list.append(sourceCam_pts)
    # save sourceCam points in list, then iterate thru imageNums with both to find intersection points
    for imNum, imPath in enumerate(target_images):
        sourcePts1 = sourceCam_pts_list[0][:, imNum, :]
        sourcePts2 = sourceCam_pts_list[1][:, imNum, :]
        
        epLines_source1 = cv2.computeCorrespondEpilines(sourcePts1, int(source_nums_in_pair[0]), fundMats[0])
        epLines_source2 = cv2.computeCorrespondEpilines(sourcePts2, int(source_nums_in_pair[1]), fundMats[1])

        epLines_source1.reshape(-1, 3)
        epLines_source2.reshape(-1, 3)
                
        intersect_point = find_epipolar_intersection(epLines_source1, epLines_source2, 
                                                     sourcePts1, sourcePts2, 
                                                     imPath, targetCam_offsets)

    # figure out how this fills in to the dataframe of the target!

    
cams, camera_pairs, source_in_pair, fundMats = retrieve_calibration(project_path_3D, source_cams, target_cams)

cfg = auxiliaryfunctions.read_config(os.path.join(project_path, 'config.yaml'))
scorer = cfg["scorer"]

source_label_files = []
for source in source_cams:
    source_label_folders = glob.glob(os.path.join(project_path, 'labeled-data', '*%s*%s' % (date_to_label, cams[source-1])))
    source_label_files.append([os.path.join(folder, "CollectedData_" + scorer + ".h5") for folder in source_label_folders])
target_label_files = []
for target in target_cams:
    target_label_folders = glob.glob(os.path.join(project_path, 'labeled-data', '*%s*%s' % (date_to_label, cams[target-1])))
    target_label_files.append([os.path.join(folder, "CollectedData_" + scorer + ".h5") for folder in target_label_folders])

for first_source_event_labels, second_source_event_labels in zip(source_label_files[0], source_label_files[1]):
    first_source_cam_name =  [cam_name for cam_name in cams if cam_name in first_source_event_labels][0]
    second_source_cam_name = [cam_name for cam_name in cams if cam_name in second_source_event_labels][0]
    for cam_target_files in target_label_files:
        for event_target_file in cam_target_files: 
            target_cam_name = [cam_name for cam_name in cams if cam_name in event_target_file][0]
            first_pair_idx = [idx for idx, pair in enumerate(camera_pairs) if 
                              pair == '%s-%s' %(first_source_cam_name, target_cam_name) or 
                              pair == '%s-%s' %(target_cam_name, first_source_cam_name)][0] 
            second_pair_idx = [idx for idx, pair in enumerate(camera_pairs) if 
                              pair == '%s-%s' %(second_source_cam_name, target_cam_name) or 
                              pair == '%s-%s' %(target_cam_name, second_source_cam_name)][0] 
            
            
            target_points = project_points([source_in_pair[first_pair_idx], source_in_pair[second_pair_idx]], 
                                           [fundMats[first_pair_idx], fundMats[second_pair_idx]], 
                                           [first_source_event_labels, second_source_event_labels],
                                           event_target_file,
                                           project_path)    
