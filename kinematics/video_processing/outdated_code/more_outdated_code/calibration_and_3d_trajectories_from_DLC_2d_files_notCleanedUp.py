import cv2
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import csv
from pathlib import Path
import scipy.io as sp
import re
import subprocess
import os
import sys
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import whiten
from tqdm import trange
import queue



############## functions ##########
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, np.transpose(v2_u)), -1.0, 1.0))

########### Class Variables ##########

class paths:
    codeDir = os.path.abspath(os.path.dirname(sys.argv[0]))

# cam1: 14
# cam2: 3, 7

######################################


def detect_ImagePoints(image_base, npy_path, nCalibFrames, nCams, nCols, nRows):

    nPointsInGrid = nCols * nRows

    goodCalibFrames = np.empty((nCams, nCalibFrames))
    grids = np.empty((nCams, nCalibFrames, nPointsInGrid, 2))

    calib_frames = range(1, nCalibFrames + 1)
    cameras = range(1, nCams + 1)
    #calib_frames = range(3, 8)
    #cameras = range(1, 2)
    for camNum in cameras:
        for frNum in calib_frames:
            
            if frNum < 10:
                image_path = '%s_cam%d_0%d.tif' % (image_base, camNum, frNum)
            else:
                image_path = '%s_cam%d_%d.tif' % (image_base, camNum, frNum)
            
            #### Set up parameters for detecting the white dots
            par = cv2.SimpleBlobDetector_Params()
    
            par.filterByArea = True
            par.minArea = 50
            par.maxArea = 1300
            par.minDistBetweenBlobs = 25 

            par.filterByColor = True
            par.blobColor = 255
    
            par.filterByInertia = True
            par.minInertiaRatio = 0.1 # 0.2 most recent change
            par.maxInertiaRatio = 1
    
            par.filterByCircularity = True
            par.minCircularity = 0.1 # also changed this from 0.1
            
            wVal = 120
            wMax = 255
            white_min = (wVal, wVal, wVal)
            white_max = (wMax, wMax, wMax)
    
            #### create blob detector
            detector = cv2.SimpleBlobDetector_create(par)
    
            #### load calib image
            image = cv2.imread(image_path)     ## This should have all points visible 
    
            #mono = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #crop_mono = mono[450:1000, 400:800]
            #avg_bright1 = np.mean(crop_mono)
            #crop_mono = mono[450:1000, 200:600]
            #avg_bright2 = np.mean(crop_mono)

            #avg_bright = max(avg_bright1, avg_bright2)
            #print(avg_bright)
    
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
            mask = cv2.inRange(blurred, white_min, white_max)
            
            avg_bright = np.mean(mask[400: 1000, 300:1100])
            avg_bright_distractors = max(np.mean(mask[0:1079, 0:300]), np.mean(mask[0:1080, 1100:1440]), np.mean(mask[0:300, 1:1439]), np.mean(mask[1000:1079, 0:1439]))
            print('avg_bright_cube = %d' % avg_bright)
            print('avg_bright_distractors = %d' % avg_bright_distractors)           

            while avg_bright_distractors < 40 or avg_bright < 3:
                if avg_bright < 5 and avg_bright_distractors < 55:
                    wVal = wVal - 2
                    white_min = (wVal, wVal, wVal)
                    mask = cv2.inRange(blurred, white_min, white_max)
                    avg_bright = np.mean(mask[400:1000, 300:1100])
                    avg_bright_distractors = max(np.mean(mask[0:1079, 0:300]), np.mean(mask[0:1079, 1100:1439]), np.mean(mask[0:300, 0:1439]), np.mean(mask[1000:1079, 0:1439]))
                    print('avg_bright_cube = %d' % avg_bright)
                    print('avg_bright_distractors = %d' % avg_bright_distractors)
                else:
                    break

            patch_bright = []
            for x_step in range(0, 4):
                for y_step in range(0, 3):
                    patch_bright.append(np.mean(mask[400+200*y_step : 400+200*(y_step+1), 300+200*x_step : 300+200*(x_step+1)]))
                    
                    print('Patch at [%d : %d, %d : %d] has patch_bright = %d' %(300+200*x_step, 300+200*(x_step+1), 400+200*y_step, 400+200*(y_step+1), patch_bright[-1]))                    
    
            if max(patch_bright) > 25:
                blurred = cv2.GaussianBlur(image, (3, 3), 0)
                mask = cv2.inRange(blurred, white_min, white_max)
                mask = cv2.erode(mask, None, iterations = 3)
                mask = cv2.dilate(mask, None, iterations = 1)
                keypoints = detector.detect(mask)                          
    
            detector = cv2.SimpleBlobDetector_create(par) 
            keypoints = detector.detect(mask)
       
            #if len(keypoints) < 2:
            #if len(keypoints) < 2 and avg_bright < 10:
             #   white_min = (60, 60, 60)
              #  white_max = (255, 255, 255)    
               # mask = cv2.inRange(blurred, white_min, white_max)    
                #keypoints = detector.detect(mask)
            #if avg_bright < 30:
            #    white_min = (60, 60, 60)
            #    white_max = (255, 255, 255)    
            #    mask = cv2.inRange(blurred, white_min, white_max)    
            #    keypoints = detector.detect(mask)                    
            #elif avg_bright > 70:
            #    blurred = cv2.GaussianBlur(image, (3, 3), 0)
            #    mask = cv2.inRange(blurred, white_min, white_max)
            #    mask = cv2.erode(mask, None, iterations = 3)
            #    mask = cv2.dilate(mask, None, iterations = 1)
            #    keypoints = detector.detect(mask)

          
            ################################
            #image_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
            #plt.imshow(image_with_keypoints, interpolation = 'bicubic') 
            #plt.show()
            #plt.imshow(mask)
            #plt.show()
            ################################    
            iters = 1
            while len(keypoints) != nCols*nRows: 
                mask = cv2.dilate(mask, None, iterations = 1) 
                par.minArea = par.minArea * 1.6
                par.maxArea = par.maxArea * 1.6
                par.minDistBetweenBlobs = par.minDistBetweenBlobs * 0.35
                par.minCircularity = par.minCircularity * 1.1
                detector = cv2.SimpleBlobDetector_create(par) 
                keypoints = detector.detect(mask) 

                ################################
                #image_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
                #plt.imshow(image_with_keypoints, interpolation = 'bicubic') 
                #plt.show()
                #plt.imshow(mask)
                #plt.show()
                ############################
    
                iters = iters + 1
                if iters == 11:
                    print('calibration image %d does not have all points visible (failed in blobDetection)' % frNum)
                    break
    
            ok, circles = cv2.findCirclesGrid(mask, (nCols, nRows), flags = (cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING), blobDetector = detector)
            image_gridDetected = cv2.drawChessboardCorners(image, (nCols, nRows), circles, ok) 
            
            plt.ion()
            plt.show()
            plt.imshow(image_gridDetected)
            plt.pause(0.001)
            
            if ok == False:
                input('The grid has not been detected. If this seems okay you, hit Enter. If not, stop the program and begin troubleshooting... ')
            else: 
                correct_arr = input('Is the arranged Right to Left, bottom to top? [y/n] ')
    
#                while correct_arr.lower() == 'n':
#                    plt.imshow(image_gridDetected)
#                    plt.pause(0.001)
#                    which_arr = input('What arrangement was detected? (e.g. - bottom to top, left to right would be entered as bt_lr): ')               
        
                while correct_arr.lower() == 'n':
                    rearr_idx = np.asarray([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

#                    elif which_arr.lower() == 'tb_rl':
#                        rearr_idx = [3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12]
#                    elif which_arr.lower() == 'lr_tb':    
#                        rearr_idx = np.arange(15, -1, -1)
#                    else: 
#                        print('\nYou must enter bt_lr, tb_rl, or lr_tb as the arrangment that was detected.')
#                        rearr_idx = np.arange(16)
             
                    circles = circles[rearr_idx]
            
                    new_image_gridDetected = cv2.drawChessboardCorners(image, (nCols, nRows), circles, ok)
                    
                    plt.imshow(new_image_gridDetected)
                    plt.pause(0.001)      
                    correct_arr = input('Is the grid correct now? (Right to Left, bottom to top) [y/n]')
            
        
            avg_unsigned_diff = []
            if ok == 1:
                v1 = circles[3] - circles[0]
                v2 = circles[7] - circles[4]
                v3 = circles[11] - circles[8]
                v4 = circles[15] - circles[12]
                for point in range(0, np.size(circles, 0) - 1): 
                    pt1 = circles[point][0] 
                    pt2 = circles[point + 1][0] 
                    diff = pt2 - pt1 
                    avg_unsigned_diff.append((abs(diff[0]) + abs(diff[1])) / 2) 
                diff_array = np.asarray(avg_unsigned_diff, 'float')
    
            if ok != 1 and iters < 11:
                print('calibration image %d does not have all points visible or could not be arranged properly (failed in findCirclesGrid)' % frNum)
                goodCalibFrames[camNum - 1, frNum - 1] = False
            elif ok == 1:
                if any(diff_array < 5): 
                    print('calibration image %d could not find a properly arranged grid (repeated points)' % frNum) 
                    goodCalibFrames[camNum - 1, frNum - 1] = False
                elif angle_between(v1, v2) > 0.7 or angle_between(v1, v3) > 0.8 or angle_between(v1, v4) > 0.9:
                    print('calibration image %d could not find a properly arranged grid (angle too large)' % frNum)
                    goodCalibFrames[camNum - 1, frNum - 1] = False
                else: 
                    grids[camNum -1, frNum - 1, :, :] = np.squeeze(circles)
                    goodCalibFrames[camNum - 1, frNum - 1] = True
    
    matchedCalibFrames = (np.where(sum(goodCalibFrames) == 2))[0]
    imagePoints = grids[:, matchedCalibFrames, :, :]
    imagePoints = imagePoints.astype('float32')
        
    #### write NPY to file ######
    imagePoints_tmpPath = '%s/imagePoints.npy' % paths.codeDir
    np.save(imagePoints_tmpPath, imagePoints)
    subprocess.run(['sudo', 'mv', imagePoints_tmpPath, npy_folder])         
          
    return imagePoints

#__________________________________________________________________________________________________________________________________________________________#

def calib_singleCam(imagePoints, objectPath, imageSize):
    
    #### Load objectPoints from csv ######
    realGrid = [] 
    with open(objectPath) as file: 
        reader = csv.reader(file) 
        for row in reader: 
            realGrid.append(row)

    realGrid = np.asarray(realGrid, dtype = 'float32') 
    
    objectPoints = np.tile(realGrid, (np.size(imagePoints, 0), 1, 1))

    ######################################
    
    ###### Intrinsic camera parameters
    
    camMat = np.asmatrix(np.array([[2000, 0, 750], [0, 2000, 575], [0, 0, 1]], dtype = 'float32')) 
    distCoeffs = np.asmatrix(np.array([-0.5, 3.0, -0.001, 0.0006, -15.0], dtype = 'float32'))

    retval, camMat, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize, camMat, distCoeffs, flags = cv2.CALIB_USE_INTRINSIC_GUESS)

    ####### If the rmse values are very high, use the following code to find the intrinsic parameters to use in future guesses. Also, change the csv file above to read *_planarPoints.csv'

    #retval, camMat, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, (1440, 1080), None, None)

    return retval, camMat, distCoeffs, rvecs, tvecs, objectPoints

#________________________________________________________________________________________________________________________________________________________#

def calib_stereo(objectPoints, imagePoints, camMats, distCoeffs, imageSize):
    
    # gcl = [1, 2, 3, 9]
    
    # retval, camMat1, distCoeffs1, camMat2, distCoeffs2, rot, trans, essential, fundamental = cv2.stereoCalibrate(objectPoints[gcl, ...], imagePoints[0,gcl,:,:], imagePoints[1,gcl,:,:], camMats[0,:,:], distCoeffs[0,:], camMats[1,:,:], distCoeffs[1,:], imageSize, flags = cv2.CALIB_FIX_INTRINSIC)

    retval, camMat1, distCoeffs1, camMat2, distCoeffs2, rot, trans, essential, fundamental = cv2.stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1], camMats[0], distCoeffs[0], camMats[1], distCoeffs[1], imageSize, flags = cv2.CALIB_FIX_INTRINSIC)

    return retval, rot, trans, essential, fundamental

def rectify_stereo(camMats, distCoeffs, imageSize, rot, trans):
    alpha = 0
    R1, R2, P1, P2 = cv2.stereoRectify(camMats[0,:,:], distCoeffs[0,:], camMats[1,:,:], distCoeffs[1,:], imageSize, rot, trans, alpha = alpha, flags=cv2.CALIB_ZERO_DISPARITY)[0:4]
    return R1, R2, P1, P2


#def triangulate_simple(DLC_undist, camMats):
#    nCams = np.size(camMats, 0)
#    A = np.zeros((nCams * 2, 4))
#    for i in range(nCams):
#        x = DLC_undist[]    

def reconstruct3D(projMats, pix_coords_reshaped, nLabels, camMats, distCoeffs, rotMats, p1, p2):
    
    trajectories = np.empty((nLabels, 3, pix_coords_reshaped[0].shape[0]))
    for partNum in range(nLabels):
        #DLC_points1 = np.transpose(pix_coords_reshaped[0][:, :, partNum])
        #DLC_points2 = np.transpose(pix_coords_reshaped[1][:, :, partNum])
        DLC_points1 = pix_coords_reshaped[0][:, :, partNum]
        DLC_points2 = pix_coords_reshaped[1][:, :, partNum]
        
        tmp1 = np.reshape(DLC_points1, [np.size(DLC_points1, 0), 1, np.size(DLC_points1, 1)])
        tmp2 = np.reshape(DLC_points2, [np.size(DLC_points2, 0), 1, np.size(DLC_points2, 1)])

#        DLC_undist1 = cv2.undistortPoints(tmp1, camMats[0,:,:], distCoeffs[0,:], None, rotMats[0, :, :], projMats[0, :-1, :])
#        DLC_undist2 = cv2.undistortPoints(tmp2, camMats[1,:,:], distCoeffs[1,:], None, rotMats[1, :, :], projMats[1, :-1, :])
#        DLC_undist1 = cv2.undistortPoints(tmp1, camMats[0,:,:], distCoeffs[0,:], None, P = projMats[0, :-1, :])
#        DLC_undist2 = cv2.undistortPoints(tmp2, camMats[1,:,:], distCoeffs[1,:], None, P = projMats[1, :-1, :])
#        DLC_undist1 = cv2.undistortPoints(tmp1, camMats[0,:,:], distCoeffs[0,:], None, R = rotMats[0])
#        DLC_undist2 = cv2.undistortPoints(tmp2, camMats[1,:,:], distCoeffs[1,:], None, R = rotMats[1])
#        DLC_undist1 = cv2.undistortPoints(tmp1, camMats[0], distCoeffs[0], None, P = camMats[0])
#        DLC_undist2 = cv2.undistortPoints(tmp2, camMats[1], distCoeffs[1], None, P = camMats[1])
        DLC_undist1 = cv2.undistortPoints(tmp1, camMats[0], distCoeffs[0])
        DLC_undist2 = cv2.undistortPoints(tmp2, camMats[1], distCoeffs[1])
#        DLC_undist1 = cv2.undistortPoints(tmp1, camMats[0,:,:], distCoeffs[0,:], None, R = rotMats[0], P = camMats[0])
#        DLC_undist2 = cv2.undistortPoints(tmp2, camMats[1,:,:], distCoeffs[1,:], None, R = rotMats[1], P = camMats[1])
        
        DLC_undist1 = DLC_undist1.squeeze()
        DLC_undist2 = DLC_undist2.squeeze()
        for pt1, pt2 in zip(DLC_undist1, DLC_undist2):
            pt1[0] = camMats[0][0, 0] * pt1[0] + camMats[0][0, 2]
            pt1[1] = camMats[0][1, 1] * pt1[1] + camMats[0][1, 2]
            pt2[0] = camMats[1][0, 0] * pt2[0] + camMats[1][0, 2]
            pt2[1] = camMats[1][1, 1] * pt2[1] + camMats[1][1, 2]

#        DLC_undist1 = np.transpose(np.reshape(DLC_undist1, [np.size(DLC_points1, 0), np.size(DLC_points1, 1)]))
#        DLC_undist2 = np.transpose(np.reshape(DLC_undist2, [np.size(DLC_points2, 0), np.size(DLC_points2, 1)]))
        
        pMats = [p1, p2]
        for frame, (pt1, pt2) in enumerate(zip(DLC_undist1, DLC_undist2)):
            points = [pt1, pt2]
            num_cams = len(camMats)
            A = np.zeros((num_cams * 2, 4))
            B = np.zeros(A.shape)
            for i in range(num_cams):
                x, y = points[i]
                pMat = pMats[i]
                A[i*2]   = x * pMat[2] - pMat[0]
                A[i*2+1] = y * pMat[2] - pMat[1]
#                for k in range(A.shape[-1]):
#                    B[i*2, k] = x * pMat[2, k] - pMat[0, k]
#                    B[i*2+1, k] = y * pMat[2, k] - pMat[1, k]
            u, s, vh = np.linalg.svd(A, full_matrices=True)
            p3d = vh[-1] 
            trajectories[partNum, :, frame] = p3d[:3] / p3d[3]
#		for (int k = 0; k < 4; k++)
#			{
#				A.at<double>(count * 2 + 0, k) = x * projMatrs.at<double>(2, k) - projMatrs.at<double>(0, k);
#				A.at<double>(count * 2 + 1, k) = y * projMatrs.at<double>(2, k) - projMatrs.at<double>(1, k);
#			}
        
#        DLC_undist1 = DLC_undist1.T
#        DLC_undist2 = DLC_undist2.T
#        stop = []
        
        # P1 = np.zeros((4,4))
        # P1[:3,:3] = [[0.91918743,0.01928893,-0.39334768],
        #               [-0.21012807,0.86876657,-0.44843154],
        #               [0.33307755,0.49484603,0.80261246]]
        # P1[:3, 3] = [-4.30049433, 1.40030131, 34.64523522]
        # P1[3, 3] = 1
        
        # P2 = np.zeros((4,4))
        # P2[:3,:3] = [[0.57832877,-0.18164672,0.79532402],
        #               [0.47728495,0.86597609,-0.14927991],
        #               [-0.66161538,0.46592905,0.58751613]]
        # P2[:3, 3] = [-7.16778854, -5.45670236, 33.93168221]
        # P2[3, 3] = 1
        
        # points_fixed = np.zeros((3, DLC_undist1.shape[-1]))     
        # num_cams = 2
        # A = np.zeros((num_cams * 2, 4))
        # # camera_mats = [P1, P2]
        # camera_mats = [projMats[0, ...], projMats[1, ...]]
        # for frame, (pts1, pts2) in enumerate(zip(DLC_undist1.T, DLC_undist2.T)):
        #     points = [pts1, pts2]
        #     for cam in range(num_cams):
        #         x, y = points[cam]
        #         mat = camera_mats[cam]
        #         A[(cam * 2):(cam * 2 + 1)] = x * mat[2] - mat[0]
        #         A[(cam * 2 + 1):(cam * 2 + 2)] = y * mat[2] - mat[1]
        #     u, s, vh = np.linalg.svd(A, full_matrices=True)
        #     p3d = vh[-1]
        #     points_fixed[:, frame] = p3d[:3] / p3d[3]
            
        
    
#        points_homog = cv2.triangulatePoints(projMats[0, :-1], projMats[1, :-1], DLC_undist1, DLC_undist2)
#        transform = np.empty((4, 4)).astype('float64')
#        transform[:3, :3] = cv2.Rodrigues(np.array([150.807, 23.163, 1.202]).astype('float64'))[0]
#        transform[:3, 3] = np.array([-7.292, -18.278, -28.870])
#        transform[3, 3] = 1
#        
#        points_homog_inv = np.linalg.inv(transform) @ points_homog
#        points_homog = transform @ points_homog
        #if partNum == 5:
            #np.savetxt('/home/daltonm/Desktop/points_homog.csv', points_homog, delimiter = ',')    
#        for frame in range(np.size(points_homog, 1)):
#            points_homog[:, frame] = points_homog[:, frame] / points_homog[3, frame]
#            points_homog_inv[:, frame] = points_homog_inv[:, frame] / points_homog_inv[3, frame]            
#
#        trajectories[partNum, :, :] = np.delete(points_homog, -1, axis = 0)
    #print(np.where(trajectories[5, :, :] < -10000))
    #np.savetxt('/home/daltonm/Desktop/traj_real.csv', trajectories[5, :, :], delimiter = ',')

        #trajectories[partNum, :, :] = cv2.convertPointsFromHomogeneous(points_homog)
        #tmp = cv2.convertPointsFromHomogeneous(points_homog)
    return trajectories   

def get_connections(xs, cam_names=None, both=True):
    n_cams = xs.shape[0]
    n_points = xs.shape[1]

    if cam_names is None:
        cam_names = np.arange(n_cams)

    connections = defaultdict(int)

    for rnum in range(n_points):
        ixs = np.where(~np.isnan(xs[:, rnum, 0]))[0]
        keys = [cam_names[ix] for ix in ixs]
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a = keys[i]
                b = keys[j]
                connections[(a,b)] += 1
                if both:
                    connections[(b,a)] += 1

    return connections


def get_calibration_graph(rtvecs, cam_names=None):
    n_cams = rtvecs.shape[0]
    n_points = rtvecs.shape[1]

    if cam_names is None:
        cam_names = np.arange(n_cams)

    connections = get_connections(rtvecs, cam_names)

    components = dict(zip(cam_names, range(n_cams)))
    edges = set(connections.items())

    # print(sorted(edges))

    graph = defaultdict(list)

    for edgenum in range(n_cams-1):
        if len(edges) == 0:
            return None

        (a, b), weight = max(edges, key=lambda x: x[1])
        graph[a].append(b)
        graph[b].append(a)

        match = components[a]
        replace = components[b]
        for k, v in components.items():
            if match == v:
                components[k] = replace

        for e in edges.copy():
            (a,b), w = e
            if components[a] == components[b]:
                edges.remove(e)

    return graph

def find_calibration_pairs(graph, source=None):
    pairs = []
    explored = set()

    if source is None:
        source = sorted(graph.keys())[0]

    q = queue.deque()
    q.append(source)

    while len(q) > 0:
        item = q.pop()
        explored.add(item)

        for new in graph[item]:
            if new not in explored:
                q.append(new)
                pairs.append( (item, new) )
    return pairs

def compute_camera_matrices(rtvecs, pairs):
    extrinsics = dict()
    source = pairs[0][0]
    extrinsics[source] = np.identity(4)
    for (a,b) in pairs:
        ext = get_transform(rtvecs, b, a)
        extrinsics[b] = np.matmul(ext, extrinsics[a])
    return extrinsics

def select_matrices(Ms):
    Ms = np.array(Ms)
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in Ms]
    tvecs = np.array([M[:3, 3] for M in Ms])
    best = get_most_common(np.hstack([rvecs, tvecs]))
    Ms_best = Ms[best]
    return Ms_best

def get_most_common(vals):
    Z = linkage(whiten(vals), 'ward')
    n_clust = max(len(vals)/10, 3)
    clusts = fcluster(Z, t=n_clust, criterion='maxclust')
    cc = Counter(clusts[clusts >= 0])
    most = cc.most_common(n=1)
    top = most[0][0]
    good = clusts == top
    return good

def mean_transform(M_list):
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in M_list]
    tvecs = [M[:3, 3] for M in M_list]

    rvec = np.mean(rvecs, axis=0)
    tvec = np.mean(tvecs, axis=0)

    return make_M(rvec, tvec)

def mean_transform_robust(M_list, approx=None, error=0.3):
    if approx is None:
        M_list_robust = M_list
    else:
        M_list_robust = []
        for M in M_list:
            rot_error = (M - approx)[:3,:3]
            m = np.max(np.abs(rot_error))
            if m < error:
                M_list_robust.append(M)
    return mean_transform(M_list_robust)

def get_transform(rtvecs, left, right):
    L = []
    for dix in range(rtvecs.shape[1]):
        d = rtvecs[:, dix]
        good = ~np.isnan(d[:, 0])

        if good[left] and good[right]:
            M_left = make_M(d[left, 0:3], d[left, 3:6])
            M_right = make_M(d[right, 0:3], d[right, 3:6])
            M = np.matmul(M_left, np.linalg.inv(M_right))
            L.append(M)
    L_best = select_matrices(L)
    M_mean = mean_transform(L_best)
    # M_mean = mean_transform_robust(L, M_mean, error=0.5)
    # M_mean = mean_transform_robust(L, M_mean, error=0.2)
    M_mean = mean_transform_robust(L, M_mean, error=0.1)
    return M_mean

def make_M(rvec, tvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(rvec)
    out[:3,:3] = rotmat
    out[:3, 3] = tvec.flatten()
    out[3, 3] = 1
    return out

def triangulate_simple(points, camera_mats, intrinsicMats):
    
    projMats = np.empty((2, 4, 4)).astype('float64')
    projMats[0, :3, :3] = cv2.Rodrigues(np.array([150.807, 23.163, 1.202]).astype('float64'))[0]
    projMats[0, :3, 3] = np.array([-7.292, -18.278, -28.870])
    projMats[0, 3, 3] = 1
    # projMats[0] = np.linalg.inv(projMats[0])
#    projMats[0, :-1] = intrinsicMats[0] @ projMats[0, :-1]
    
    projMats[1, :3, :3] = cv2.Rodrigues(np.array([165.744, -52.686, -17.437]).astype('float64'))[0]
    projMats[1, :3, 3] = np.array([29.199, -12.386, -15.049])
    projMats[1, 3, 3] = 1
    # projMats[1] = np.linalg.inv(projMats[1])
#    projMats[1, :-1] = intrinsicMats[1] @ projMats[1, :-1]
    
    camera_mats = projMats
    
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d_inv = np.linalg.inv(projMats[1]) @ p3d
    p3d = projMats[1] @ p3d
    
    p3d = p3d / p3d[3]
    p3d_inv = p3d_inv / p3d_inv[3]
    print(p3d) 
    print(p3d_inv)
    # p3d_world = np.linalg.inv(projMats[0]) @ p3d
    p3d_world = projMats[0] @ p3d
    p3d_world = p3d_world[:3]
    # p3d = p3d[:3] / p3d[3]

    return p3d

def triangulate(points, rotMats, projMats, camMats, distCoeffs, undistort=True, progress=False):
    """Given an CxNx2 array, this returns an Nx3 array of points,
    where N is the number of points and C is the number of cameras"""

    one_point = False
    if len(points.shape) == 2:
        points = points.reshape(-1, 1, 2)
        one_point = True

    if undistort:
        new_points = np.empty(points.shape)
        for cnum, (camMat, distCo, rot, proj) in enumerate(zip(camMats, distCoeffs, rotMats, projMats)):
            # must copy in order to satisfy opencv underneath
            sub = np.copy(points[cnum])
            new_points[cnum] = undistort_points(sub, camMat, distCo, rot, proj)
        points = new_points

    n_cams, n_points, _ = points.shape

    out = np.empty((n_points, 3))
    out[:] = np.nan

    cam_mats = projMats

    if progress:
        iterator = trange(n_points, ncols=70)
    else:
        iterator = range(n_points)

    for ip in iterator:
        subp = points[:, ip, :]
        good = ~np.isnan(subp[:, 0])
        if np.sum(good) >= 2:
            out[ip] = triangulate_simple(subp[good], cam_mats[good], camMats)

    if one_point:
        out = out[0]

    return out

def undistort_points(points, camMat, distCo, rot, proj):
    shape = points.shape
    points = points.reshape(-1, 1, 2)
    out = cv2.undistortPoints(points, camMat.astype('float64'), distCo.astype('float64'), None, R = rot.astype('float64'), P = camMat.astype('float64'))
#    out = cv2.undistortPoints(points, camMat.astype('float64'), distCo.astype('float64'), None, R = rot.astype('float64'), P = proj[:3, :3].astype('float64'))
#    out = cv2.undistortPoints(points, camMat.astype('float64'), distCo.astype('float64'), None, P = camMat.astype('float64'))
#    out = cv2.undistortPoints(points, camMat.astype('float64'), distCo.astype('float64'), None, P = proj[:3, :3].astype('float64'))
#    out = cv2.undistortPoints(points, camMat.astype('float64'), distCo.astype('float64'), R = rot.astype('float64'))
#    out = cv2.undistortPoints(points, camMat.astype('float64'), distCo.astype('float64'))

    return out.reshape(shape)

##################################################################################################################################################################
#################################################################################################################################################################

def main():
 
#####################

    session_date = '2019_04_15'
    marm = 'Pat'
    # dlc_projectName = 'Pat_XROMM_relabel_closer_to_joints_test-Dalton-2020-05-11'
    #                   #'validation_Pat-Dalton-2020-01-23'
    #                   #'validation_Tony-Dalton-2020-01-05' 
    #                   #'Pat_XROMM_relabel_closer_to_joints_test-Dalton-2020-05-11'
    dlc_resultsPattern = 'Jan23shuffle1_110000' 
                        #'May11shuffle1_140000'
                        #'Jan5shuffle1_60000'
                        #'Jan23shuffle1_110000' 
    if session_date == '2019_04_14':
        image_for_triangulation = 2
    elif session_date == '2019_04_15':
        image_for_triangulation = 1
    anipose = False

#####################

    opSys = 'windows' # 'linux'

    if opSys == 'windows':
        base = r'Z:/'
    elif opSys == 'linux':
        base = '/media/CRI/'

    npy_folder = os.path.join(base, 'marmosets/XROMM_and_RGB_sessions/RGB_videos/%s/pre_calib' % session_date)
    objectPath = os.path.join(base, 'marmosets/calibration_reference_files/Lego_cube_4x5_objectPoints.csv')
    # DLC_base = '/home/marmosets/Documents/dlc_local/%s/videos' % dlc_projectName 
    # DLC_base = os.path.join(base, 'marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/joint_based_labels') #'/media/CRI/marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/' #'/home/marmosets/Documents/dlc_local/videos/Pat_videos'
    # traj_dir = os.path.join(base, 'marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/joint_based_labels', str(session_date) + '_calib_image_set_' + str(image_for_triangulation))
    DLC_base = os.path.join(base, 'marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/post_first_refinement') #'/media/CRI/marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/' #'/home/marmosets/Documents/dlc_local/videos/Pat_videos'
    traj_dir = os.path.join(base, 'marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/corrected_calibration_post_first_refinement', str(session_date) + '_calib_image_set_' + str(image_for_triangulation))
               #'/media/CRI/marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/anipose' 
    # os.makedirs(traj_dir)

    if anipose:
        print('fix paths for anipose in this section near top of main()')
        # cam1_files = sorted(glob.glob(os.path.join('/home/marmosets/Documents/anipose_XROMM_pat/anipose_thresh_equals_0.2/' + session_date, marm,  session_date + '*cam1*.h5')))
        # cam2_files = sorted(glob.glob(os.path.join('/home/marmosets/Documents/anipose_XROMM_pat/anipose_thresh_equals_0.2/' + session_date, marm,  session_date + '*cam2*.h5')))
    else:
        cam1_files = sorted(glob.glob(os.path.join(DLC_base, session_date + '*cam1*' + dlc_resultsPattern + '.h5')))
        cam2_files = sorted(glob.glob(os.path.join(DLC_base, session_date + '*cam2*' + dlc_resultsPattern + '.h5')))

    for fNum in range(len(cam1_files)):
        pix_coords = [0, 0]
        pix_coords[0] = np.array(pd.read_hdf(cam1_files[fNum], 'df_with_missing')).astype(float)
        pix_coords[1] = np.array(pd.read_hdf(cam2_files[fNum], 'df_with_missing')).astype(float)  
        
        nLabels = int(np.size(pix_coords[0], 1) / 3)

        pix_coords_reshaped = [0, 0]
        for camNum in range(len(pix_coords)):
            pix_traj_array = np.empty((np.size(pix_coords[0], 0), 2, nLabels))
            for labelNum in range(nLabels):
                pix_traj_array[:, :, labelNum] = pix_coords[camNum][:, 3*labelNum : 3*(labelNum+1) - 1]     
            pix_coords_reshaped[camNum] = pix_traj_array

        nCams = 2
        if session_date == '2019_04_14':
            nCalibFrames = 7
        elif session_date == '2019_04_15':
            nCalibFrames = 18
        nCols = 4
        nRows = 5
        imageSize = (1440, 1080)
    
        rmse = np.empty((nCams))
        camMats = np.empty((nCams, 3, 3))
        distCoeffs = np.empty((nCams, 5))
        rvecs = np.empty((nCams, nCalibFrames, 3, 1))
        tvecs = np.empty((nCams, nCalibFrames, 3, 1))
        projMats = np.empty((nCams, 3, 4))
        rotMats = np.empty((nCams, 3, 3))
    
        npy_path = '%s/imagePoints.npy' %(npy_folder) 
        config = Path(npy_path) 
        if config.is_file():
            imagePoints = np.load(npy_path)
        else:   
            imagePoints = detect_ImagePoints(image_base, npy_path, nCalibFrames, nCams, nCols, nRows)
    
        if np.size(imagePoints, 1) < nCalibFrames:
            rvecs = rvecs[:, 0:np.size(imagePoints, 1), :]
            tvecs = tvecs[:, 0:np.size(imagePoints, 1), :]
    
        cameras = range(1, nCams + 1)
        for cam in cameras:
            rmse[cam - 1], camMats[cam - 1, :, :], distCoeffs[cam - 1, :], rvecs[cam - 1, :, :, :], tvecs[cam - 1, :, :, :], objectPoints = calib_singleCam(imagePoints[cam - 1, :, :, :], objectPath, imageSize) 
        
        # print('\n rmse (reprojection error) in single camera calibration step: \n cam1 = %f pixels \n cam2 = %f pixels' % (rmse[0], rmse[1]))
    
        rmse_stereo, rot, trans, essential, fundamental = calib_stereo(objectPoints, imagePoints, camMats, distCoeffs, imageSize)
        
        # print('rot, trans')
        # rotvec, tmp = cv2.Rodrigues(rot)
        # print((rotvec, trans))
    
        print('stereo_rmse = %f pixels' % rmse_stereo)

        rotMats[0, :, :], rotMats[1, :, :], projMats[0, :, :], projMats[1, :, :] = rectify_stereo(camMats, distCoeffs, imageSize, rot, trans)
        event = re.findall('event[0-9]{3}', cam1_files[fNum])[0]
        # print(event)
        if event == 'event018':
            catch = []
        
        # rtvecs = np.empty((n_cams, n_detects, 6), dtype='float64')
        anipose = False
        if anipose:
            rtvecs = np.dstack((rvecs.squeeze(), tvecs.squeeze())) 
            graph = get_calibration_graph(rtvecs)
            pairs = find_calibration_pairs(graph, source=0)
            extrinsics = compute_camera_matrices(rtvecs, pairs)
            projMats = np.empty((2, 4, 4))
            projMats[0] = extrinsics[0]
            projMats[1] = extrinsics[1]     
            
            points_2d = np.empty((len(pix_coords_reshaped), pix_coords_reshaped[0].shape[0], pix_coords_reshaped[0].shape[2], pix_coords_reshaped[0].shape[1]))
            points_2d[0] = np.swapaxes(pix_coords_reshaped[0], 1, 2)
            points_2d[1] = np.swapaxes(pix_coords_reshaped[1], 1, 2)
            n_cams, n_frames, n_joints, _ = points_2d.shape
            points_shaped = points_2d.reshape(n_cams, n_frames*n_joints, 2)
            points_3d = triangulate(points_shaped, rotMats, projMats, camMats, distCoeffs, undistort=True, progress=True)
            points_3d = points_3d.reshape((n_frames, n_joints, 3))
            
        else:

            projMats = np.hstack((projMats, np.zeros((2, 1, 4))))
            projMats[:, -1, -1] = 1
            # projMats = np.empty((2, 4, 4))
            # projMats[0, :3, :3] = [[0.91918743,0.01928893,-0.39334768],
            #                        [-0.21012807,0.86876657,-0.44843154],
            #                        [0.33307755,0.49484603,0.80261246]]
            # projMats[0, :3, 3] = [-4.30049433, 1.40030131, 34.64523522]
            # projMats[0, 3, 3] = 1
            
            # projMats[1, :3, :3] = [[0.57832877,-0.18164672,0.79532402],
            #                        [0.47728495,0.86597609,-0.14927991],
            #                        [-0.66161538,0.46592905,0.58751613]]
            # projMats[1, :3, 3] = [-7.16778854, -5.45670236, 33.93168221]
            # projMats[1, 3, 3] = 1

            # projMats = np.empty((2, 4, 4)).astype('float64')
            # projMats[0, :3, :3] = cv2.Rodrigues(np.array([150.807, 23.163, 1.202]).astype('float64'))[0]
            # projMats[0, :3, 3] = np.array([-7.292, -18.278, -28.870])
            # projMats[0, 3, 3] = 1
            # projMats[0] = np.linalg.inv(projMats[0])
            
            # projMats[1, :3, :3] = cv2.Rodrigues(np.array([165.744, -52.686, -17.437]).astype('float64'))[0]
            # projMats[1, :3, 3] = np.array([29.199, -12.386, -15.049])
            # projMats[1, 3, 3] = 1
            # projMats[1] = np.linalg.inv(projMats[1])
            
            pick_best = False
            pMat1 = []
            pMat2 = []
            if pick_best:
                rtvecs = np.dstack((rvecs.squeeze(), tvecs.squeeze()))
                pairs = [(0, 1)]
                best = get_most_common(rtvecs[0])
                if session_date == '2019_04_14':
                    best = ~best
                
                for imageNum in range(len(best)):
                    rotMat1, tmp = cv2.Rodrigues(rvecs[0, imageNum])
                    rotMat2, tmp = cv2.Rodrigues(rvecs[1, imageNum])
                    tvec1 = tvecs[0, imageNum]
                    tvec2 = tvecs[1, imageNum]
                    
                    pMat1.append(np.vstack((camMats[0] @ np.hstack((rotMat1, tvec1)), [0, 0, 0, 1])))
                    pMat2.append(np.vstack((camMats[1] @ np.hstack((rotMat2, tvec2)), [0, 0, 0, 1])))    
                
                best1 = [mat for i, mat in enumerate(pMat1) if best[i]]
                best2 = [mat for i, mat in enumerate(pMat2) if best[i]]
                best1 = np.array(best1)
                best2 = np.array(best2)
                M1_mean = mean_transform(best1)
                M2_mean = mean_transform(best2)
                
                M1_mean = mean_transform_robust(pMat1, M1_mean, error=0.3)
                M2_mean = mean_transform_robust(pMat2, M2_mean, error=0.3)
                reconstructedTrajectories = reconstruct3D(projMats, pix_coords_reshaped, nLabels, camMats, distCoeffs, rotMats, M1_mean, M2_mean)

                
            else:
                imageNum = image_for_triangulation
                rotMat1, tmp = cv2.Rodrigues(rvecs[0, imageNum])
                rotMat2, tmp = cv2.Rodrigues(rvecs[1, imageNum])
                tvec1 = tvecs[0, imageNum]
                tvec2 = tvecs[1, imageNum]
                
                pMat1_fromXMA = camMats[0] @ np.hstack((rotMat1, tvec1))
                pMat2_fromXMA = camMats[1] @ np.hstack((rotMat2, tvec2))
                reconstructedTrajectories = reconstruct3D(projMats, pix_coords_reshaped, nLabels, camMats, distCoeffs, rotMats, pMat1_fromXMA, pMat2_fromXMA)
                if fNum == 0:
                    nLabels = 1
                    
                    if session_date == '2019_04_14':
                        cam1_points = np.array([[856 , 626], 
                                                [1202, 627],
                                                [1317, 386]], dtype = np.float32)
                        cam2_points = np.array([[977 , 622],
                                                [1050, 775],
                                                [1128, 602]], dtype = np.float32)
                    elif session_date == '2019_04_15':
                        cam1_points = np.array([[787 , 588], 
                                                [1119, 581],
                                                [1226, 347]], dtype = np.float32)
                        cam2_points = np.array([[836 , 629],
                                                [908 , 792],
                                                [998 , 618]], dtype = np.float32)
                        
                    cam1_points = np.reshape(cam1_points, (cam1_points.shape[0],cam1_points.shape[1], 1))
                    cam2_points = np.reshape(cam2_points, (cam2_points.shape[0],cam2_points.shape[1], 1))
                    basis_points = [cam1_points, cam2_points]
                    reconstructedBasis = reconstruct3D(projMats, basis_points, nLabels, camMats, distCoeffs, rotMats, pMat1_fromXMA, pMat2_fromXMA)
                
                    if opSys == 'windows':
                        traj_path = '%s/%s_refFrame_corrected.npy' % (traj_dir, session_date)
                        np.save(traj_path, reconstructedBasis[0])
                    elif opSys == 'linux':
                        traj_tmpPath = '%s/%s_refFrame_corrected.npy' % (paths.codeDir, session_date)
                        np.save(traj_tmpPath, reconstructedBasis[0])
                        subprocess.run(['sudo', 'mv', traj_tmpPath, traj_dir])
                    
    
        
        # projMats = projMats[:, :-1]
        # reconstructedTrajectories = np.swapaxes(points_3d, 0, 2)
        # reconstructedTrajectories = np.swapaxes(reconstructedTrajectories, 0,1)

        event = re.findall('event[0-9]{3}', cam1_files[fNum])[0]
        session = re.findall('session[0-9]', cam1_files[fNum])[0]
        
        if event != 'event023' and event != 'event026':        
        
            if opSys == 'windows':
                traj_path = '%s/%s_%s_%s_%s_trajectories_%s.npy' % (traj_dir, session_date, session, marm, event, dlc_resultsPattern)
                np.save(traj_path, reconstructedTrajectories)
            elif opSys == 'linux':
                traj_tmpPath = '%s/%s_%s_%s_%s_trajectories_%s.npy' % (paths.codeDir, session_date, session, marm, event, dlc_resultsPattern)
        
                np.save(traj_tmpPath, reconstructedTrajectories)
                subprocess.run(['sudo', 'mv', traj_tmpPath, traj_dir])
                print(traj_dir)
     
    imageNum = 2

    rvecMat1, tmp = cv2.Rodrigues(rvecs[0, imageNum, ...])
    rvecMat2, tmp = cv2.Rodrigues(rvecs[1, imageNum, ...])
    
    print('Cam mats')
    print(camMats[0, ...])
    print(camMats[1, ...])
    
    print('\n Dist coeffs')
    print(distCoeffs[0, ...])
    print(distCoeffs[1, ...]) 
    
    print('\n rvecs')
    print(list(rvecs[0, imageNum, ...].flatten()))
    print(list(rvecs[1, imageNum, ...].flatten()))
    
    print('\n Tvecs')
    print(list(tvecs[0, imageNum].flatten()))
    print(list(tvecs[1, imageNum].flatten()))
    
	# adjust y - inversion
    # trans1 = tvecs[0, imageNum, ...]
    # trans2 = tvecs[1, imageNum, ...]
    # for trans in [trans1, trans2]:
    #     trans[0] = -trans[0]
    #     trans[2] = -trans[2]
        
        

    # 	for (int i = 0; i < 3; i ++)
    # 	{
    # 		rotTmp.at<double>(0, i) = -rotTmp.at<double>(0, i);
    # 		rotTmp.at<double>(2, i) = -rotTmp.at<double>(2, i);
    # 	}
    # 	camTmp.at<double>(1, 2) = (getHeight() - 1) - camTmp.at<double>(1, 2);

    # print('\nrVecs\n')
    # print(rvecMat1)
    # print(rvecMat2)
    # # print(rvecs[0, ...])      
    # # print(rvecs[1, ...])
    # # print('\navg rvec\n')
    # # print(np.mean(rvecs[0, ...], 0)) 
    # # print(np.mean(rvecs[1, ...], 0))         
    # print('\ntVecs\n')        
    # print(tvecs[0, imageNum, ...])      
    # print(tvecs[1, imageNum, ...]) 
    # print(tvecs[0,  ...])      
    # print(tvecs[1,  ...]) 
    # # print('\navg tvec\n')
    # # print(np.mean(tvecs[0, ...], 0)) 
    # # print(np.mean(tvecs[1, ...], 0))      
    # print('\ncamMats\n')  
    # print(camMats[0, ...])
    # print(camMats[1, ...])
    # print('\ndistCoeffs\n')
    # print(distCoeffs[0, ...])
    # print(distCoeffs[1, ...]) 
    # # print('\nprojection mats\n')
    # # print(projMats[0, ...])
    # # print(projMats[1, ...])
    
    # camde, rotde, transde = cv2.decomposeProjectionMatrix(projMats[0, ...])[:3]
    # rotvecde, tmp = cv2.Rodrigues(rotde)

    # camde2, rotde2, transde2 = cv2.decomposeProjectionMatrix(projMats[1, ...])[:3]
    # rotvecde2, tmp = cv2.Rodrigues(rotde2)
    # print((camde, camde2))
    # print((rotde, rotde2))
    # print((rotvecde, rotvecde2))
    # print((transde, transde2))
    #%% Project 3D origin and axes onto DLC images (04_15, calib_Image=1)

    origin_and_axes = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype = np.float32)
    imagePoints_cam1, tmp = cv2.projectPoints(origin_and_axes, rvecs[0, 1], tvecs[0, 1], camMats[0], distCoeffs[0])
    imagePoints_cam2, tmp = cv2.projectPoints(origin_and_axes, rvecs[1, 1], tvecs[1, 1], camMats[1], distCoeffs[1])

    # print(imagePoints_cam1)
    # print(imagePoints_cam2)

if __name__ == '__main__':
    main()

#%% computing origin for refFrame in corrected DLC - incomplete/idk where I was going with this


dates = ['2019_04_14', '2019_04_15']
cam1_origin = np.array([[877, 609],[809, 570]], dtype = np.float32)
cam2_origin = np.array([[980, 618],[838, 624]], dtype = np.float32)


#%% Project 3D origin and axes onto DLC images (04_15, calib_Image=1)

# origin_and_axes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = np.float32)

# imagePoints_cam1, tmp = cv2.projectPoints(origin_and_axes, rvecs[0, 1], tvecs[0, 1], camMats[0], distCoeffs[0])


#%%
################### Need to extend this to run thru all files in a directory, then store all sets of matched calibration images that had grids detected. THen run thru following steps

# 1) DONE cv2.calibrateCameras(ObjectPoints, grids, etc) for each camera

# 2) DONE cv2.stereoCalibrate with flags = CALIB_FIX_INRINSIC

# 3) DONE cv2.stereoRectify

# 4) cv2.triangulatePoints  


# points = np.loadtxt(open("/media/CRI/marmosets/XROMM_and_RGB_sessions/XROMM_videos/20190415_fullRecording/2019-04-15_11-47_Evt18/Event18_labeled/Event_18/data/Marker010points2d.csv", "rb"), delimiter=",").astype(np.float32)
# camMat1 = np.loadtxt(open("/media/CRI/marmosets/XROMM_and_RGB_sessions/XROMM_videos/20190415_fullRecording/2019-04-15_11-47_Evt18/Event18_labeled/Camera 1/data/Camera 1_CameraMatrix.csv", "rb"), delimiter=",").astype(np.float32)
# camMat2 = np.loadtxt(open("/media/CRI/marmosets/XROMM_and_RGB_sessions/XROMM_videos/20190415_fullRecording/2019-04-15_11-47_Evt18/Event18_labeled/Camera 2/data/Camera 2_CameraMatrix.csv", "rb"), delimiter=",").astype(np.float32)

# cam1_calibPath = '/media/CRI/marmosets/XROMM_and_RGB_sessions/XROMM_videos/20190415_fullRecording/2019-04-15_11-47_Evt18/Event18_labeled/Camera 1/data'
# cam2_calibPath = '/media/CRI/marmosets/XROMM_and_RGB_sessions/XROMM_videos/20190415_fullRecording/2019-04-15_11-47_Evt18/Event18_labeled/Camera 2/data' 

# imPts1 = np.zeros((8, 68, 2)).astype(np.float32)
# imPts2 = np.zeros_like(imPts1)
# for cNum, (f1, f2) in enumerate(zip(sorted(glob.glob(os.path.join(cam1_calibPath, '*PointsDetected.csv'))), sorted(glob.glob(os.path.join(cam2_calibPath, '*PointsDetected.csv'))))):
#     imPts1[cNum, ...] = np.loadtxt(open(f1), delimiter=",").astype(np.float32)
#     imPts2[cNum, ...] = np.loadtxt(open(f2), delimiter=",").astype(np.float32)
    
# objPts_tmp = np.loadtxt(open('/media/CRI/marmosets/XROMM_and_RGB_sessions/Reference_and_template_files/small_lego_cube_v1_framespec_cm.csv', 'rb'), delimiter=',', skiprows = 1).astype(np.float32)

# objPts = np.tile(objPts_tmp, (8, 1, 1))


# xromm1 = points[:, :2].T
# xromm2 = points[:, 2:].T

# distCoeffs1 = np.array([0,0,0,0,0]).astype(np.float32)
# distCoeffs2 = np.array([0,0,0,0,0]).astype(np.float32)
# imageSize = (900,900)

        
        
# #########

# retval, camMat1, distCoeffs1, camMat2, distCoeffs2, rot, trans, essential, fundamental = cv2.stereoCalibrate(objPts, 
#                                                                                                              imPts1, imPts2, 
#                                                                                                              camMat1, distCoeffs1, 
#                                                                                                              camMat2, distCoeffs2, 
#                                                                                                              imageSize, 
#                                                                                                              flags = cv2.CALIB_FIX_INTRINSIC)
# R1, R2, P1, P2 = cv2.stereoRectify(camMat1, distCoeffs1, camMat2, distCoeffs2, imageSize, rot, trans)[0:4]

# points_homog = cv2.triangulatePoints(P1, P2, xromm1, xromm2)
# #if partNum == 5:
#     #np.savetxt('/home/daltonm/Desktop/points_homog.csv', points_homog, delimiter = ',')    
# for frame in range(np.size(points_homog, 1)):
#     points_homog[:, frame] = points_homog[:, frame] / points_homog[3, frame]

# A = np.zeros((4, 4))
# points_fixed = np.zeros((3, xromm1.shape[-1]))
# for frame, (pts1, pts2) in enumerate(zip(xromm1.T, xromm2.T)):
#     for k in range(4):
#         A[0, k] = pts1[0] * P1[2, k] - P1[0, k]
#         A[1, k] = pts1[1] * P1[2, k] - P1[1, k]
#         A[2, k] = pts2[0] * P2[2, k] - P2[0, k]
#         A[3, k] = pts2[1] * P2[2, k] - P2[1, k]

#     points_homog = np.linalg.svd(A, False)[1] # cv2.SVDecomp(A)[0]
#     # points_homog = points_homog / points_homog[3]
#     points_fixed[:, frame] = points_homog[:-1].flatten()