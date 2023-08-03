import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import csv
from pathlib import Path
import re
import shutil
import os
import sys
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import whiten
from tqdm import trange
import queue

class paths:
    codeDir = os.path.abspath(os.path.dirname(sys.argv[0]))

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, np.transpose(v2_u)), -1.0, 1.0))

def detect_ImagePoints(image_base, npy_path, nCalibFrames, nCams, nCols, nRows):

    nPointsInGrid = nCols * nRows

    goodCalibFrames = np.empty((nCams, nCalibFrames))
    grids = np.empty((nCams, nCalibFrames, nPointsInGrid, 2))

    calib_frames = range(1, nCalibFrames + 1)
    cameras = range(1, nCams + 1)
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
   
            iters = 1
            while len(keypoints) != nCols*nRows: 
                mask = cv2.dilate(mask, None, iterations = 1) 
                par.minArea = par.minArea * 1.6
                par.maxArea = par.maxArea * 1.6
                par.minDistBetweenBlobs = par.minDistBetweenBlobs * 0.35
                par.minCircularity = par.minCircularity * 1.1
                detector = cv2.SimpleBlobDetector_create(par) 
                keypoints = detector.detect(mask) 
    
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
        
                while correct_arr.lower() == 'n':
                    rearr_idx = np.asarray([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
             
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
    shutil.move(imagePoints_tmpPath, npy_path)         
          
    return imagePoints

def calib_singleCam(imagePoints, objectPath, imageSize):
    
    # Load objectPoints from csv
    realGrid = [] 
    with open(objectPath) as file: 
        reader = csv.reader(file) 
        for row in reader: 
            realGrid.append(row)

    realGrid = np.asarray(realGrid, dtype = 'float32') 
    objectPoints = np.tile(realGrid, (np.size(imagePoints, 0), 1, 1))
    
    # Intrinsic camera parameters initialized    
    camMat = np.asmatrix(np.array([[2000, 0, 750], [0, 2000, 575], [0, 0, 1]], dtype = 'float32')) 
    distCoeffs = np.asmatrix(np.array([-0.5, 3.0, -0.001, 0.0006, -15.0], dtype = 'float32'))

    retval, camMat, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize, camMat, distCoeffs, flags = cv2.CALIB_USE_INTRINSIC_GUESS)

    return retval, camMat, distCoeffs, rvecs, tvecs, objectPoints

def calib_stereo(objectPoints, imagePoints, camMats, distCoeffs, imageSize):
    retval, camMat1, distCoeffs1, camMat2, distCoeffs2, rot, trans, essential, fundamental = cv2.stereoCalibrate(objectPoints, 
                                                                                                                 imagePoints[0], 
                                                                                                                 imagePoints[1], 
                                                                                                                 camMats[0], 
                                                                                                                 distCoeffs[0], 
                                                                                                                 camMats[1], 
                                                                                                                 distCoeffs[1], 
                                                                                                                 imageSize, 
                                                                                                                 flags = cv2.CALIB_FIX_INTRINSIC)
    return retval, rot, trans, essential, fundamental

def rectify_stereo(camMats, distCoeffs, imageSize, rot, trans):
    alpha = 0
    R1, R2, P1, P2 = cv2.stereoRectify(camMats[0,:,:], 
                                       distCoeffs[0,:], 
                                       camMats[1,:,:], 
                                       distCoeffs[1,:], 
                                       imageSize, 
                                       rot, 
                                       trans, 
                                       alpha = alpha, 
                                       flags=cv2.CALIB_ZERO_DISPARITY)[0:4]
    return R1, R2, P1, P2   

def reconstruct3D(projMats, pix_coords_reshaped, nLabels, camMats, distCoeffs, rotMats, p1, p2):
    
    trajectories = np.empty((nLabels, 3, pix_coords_reshaped[0].shape[0]))
    for partNum in range(nLabels):
        DLC_points1 = pix_coords_reshaped[0][:, :, partNum]
        DLC_points2 = pix_coords_reshaped[1][:, :, partNum]
        
        tmp1 = np.reshape(DLC_points1, [np.size(DLC_points1, 0), 1, np.size(DLC_points1, 1)])
        tmp2 = np.reshape(DLC_points2, [np.size(DLC_points2, 0), 1, np.size(DLC_points2, 1)])

        DLC_undist1 = cv2.undistortPoints(tmp1, camMats[0], distCoeffs[0])
        DLC_undist2 = cv2.undistortPoints(tmp2, camMats[1], distCoeffs[1])
       
        DLC_undist1 = DLC_undist1.squeeze()
        DLC_undist2 = DLC_undist2.squeeze()
        for pt1, pt2 in zip(DLC_undist1, DLC_undist2):
            pt1[0] = camMats[0][0, 0] * pt1[0] + camMats[0][0, 2]
            pt1[1] = camMats[0][1, 1] * pt1[1] + camMats[0][1, 2]
            pt2[0] = camMats[1][0, 0] * pt2[0] + camMats[1][0, 2]
            pt2[1] = camMats[1][1, 1] * pt2[1] + camMats[1][1, 2]
        
        pMats = [p1, p2]
        for frame, (pt1, pt2) in enumerate(zip(DLC_undist1, DLC_undist2)):
            points = [pt1, pt2]
            num_cams = len(camMats)
            A = np.zeros((num_cams * 2, 4))
            for i in range(num_cams):
                x, y = points[i]
                pMat = pMats[i]
                A[i*2]   = x * pMat[2] - pMat[0]
                A[i*2+1] = y * pMat[2] - pMat[1]
            u, s, vh = np.linalg.svd(A, full_matrices=True)
            p3d = vh[-1] 
            trajectories[partNum, :, frame] = p3d[:3] / p3d[3]
    return trajectories   

def main():
 
    session_date = '2019_04_14'
    marm = 'Pat'
    # dlc_projectName = 'Pat_XROMM_relabel_closer_to_joints_test-Dalton-2020-05-11'
    #                   #'validation_Pat-Dalton-2020-01-23'
    #                   #'validation_Tony-Dalton-2020-01-05' 
    #                   #'Pat_XROMM_relabel_closer_to_joints_test-Dalton-2020-05-11'
    dlc_resultsPattern = 'Jan5shuffle1_60000' 
                        #'May11shuffle1_140000'
                        #'Jan5shuffle1_60000'
                        #'Jan23shuffle1_110000' 
    if session_date == '2019_04_14':
        image_for_triangulation = 2
    elif session_date == '2019_04_15':
        image_for_triangulation = 1

    opSys = 'windows' # 'linux'

    if opSys == 'windows':
        base = r'Z:/'
    elif opSys == 'linux':
        base = '/media/CRI/'

    npy_folder = os.path.join(base, 'marmosets/XROMM_and_RGB_sessions/RGB_videos/%s/pre_calib' % session_date)
    objectPath = os.path.join(base, 'marmosets/calibration_reference_files/Lego_cube_4x5_objectPoints.csv')
    DLC_base = os.path.join(base, 'marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/post_first_refinement') #'/media/CRI/marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/' #'/home/marmosets/Documents/dlc_local/videos/Pat_videos'
    traj_dir = os.path.join(base, 'marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/corrected_calibration_post_first_refinement', str(session_date) + '_calib_image_set_' + str(image_for_triangulation))

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
            image_base = os.path.join(npy_folder, session_date)
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
    
        projMats = np.hstack((projMats, np.zeros((2, 1, 4))))
        projMats[:, -1, -1] = 1
                    
        imageNum = image_for_triangulation
        rotMat1, tmp = cv2.Rodrigues(rvecs[0, imageNum])
        rotMat2, tmp = cv2.Rodrigues(rvecs[1, imageNum])
        tvec1 = tvecs[0, imageNum]
        tvec2 = tvecs[1, imageNum]
        
        pMat1_fromXMA = camMats[0] @ np.hstack((rotMat1, tvec1))
        pMat2_fromXMA = camMats[1] @ np.hstack((rotMat2, tvec2))
        reconstructedTrajectories = reconstruct3D(projMats, pix_coords_reshaped, nLabels, camMats, distCoeffs, rotMats, pMat1_fromXMA, pMat2_fromXMA)

    rvecMat1, tmp = cv2.Rodrigues(rvecs[0, imageNum, ...])
    rvecMat2, tmp = cv2.Rodrigues(rvecs[1, imageNum, ...])
    
    print((type(camMats), type(camMats[0])))
    print(type(rvecs[0]))
    print(type(distCoeffs[0]))
    
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

    #%% Project 3D origin and axes onto DLC images (04_15, calib_Image=1)

    origin_and_axes = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype = np.float32)
    imagePoints_cam1, tmp = cv2.projectPoints(origin_and_axes, rvecs[0, 1], tvecs[0, 1], camMats[0], distCoeffs[0])
    imagePoints_cam2, tmp = cv2.projectPoints(origin_and_axes, rvecs[1, 1], tvecs[1, 1], camMats[1], distCoeffs[1])

if __name__ == '__main__':
    main()

