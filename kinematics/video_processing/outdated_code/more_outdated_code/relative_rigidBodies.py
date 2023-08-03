import numpy as np
#import cv2
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import glob
from pyquaternion import Quaternion

###### DIRECTION = 1 seems to be the right direction for maya. Now I just need to figure 
###### out why the parts seems to flipped, roughly in the y-dir. Also, need to use the maya file
###### that has only trial 18 imported for the xromm

rigidBody = 'torso'
direction = 1 
ct_path = 'Z:/marmosets/XROMM_and_RGB_sessions/new_skin_and_bones_objects_pat/marker_coordinates/coords_matched_to_xromm_%s.csv' % rigidBody
# ct_path = 'C:/Users/daltonm/Documents/Lab_Files/rigid_body_computation/exported_marker_coordinates/joint_based_coords_%s.csv' % rigidBody
           #'Z:/marmosets/XROMM_and_RGB_sessions/RGB_videos/Pat_maya_files/markers_fixed_2019_05_03/joint_based_coords_%s.csv' % rigidBody

traj_paths = sorted(glob.glob(r'Z:/marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/corrected_calibration_post_first_refinement/2019_04_15_session1_Pat_event018_*110000*.npy'))
# traj_paths = sorted(glob.glob(r'Z:/marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/joint_based_labels/2019_04_15_calib_image_set_1/2019_04_15_session1_Pat*140000*.npy'))
             #sorted(glob.glob(r'Z:/marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/2019_04_15_session1_Pat*150000*.npy'))

ct_coordinates = pd.read_csv(ct_path)

traj_coordinates = np.load(traj_paths[0]) #np.array(loadmat(traj_paths[0])['trajectories'])[..., 9:]
for path in traj_paths[1:]:
    traj_coordinates = np.dstack((traj_coordinates, np.load(path)))

# traj_path = 'Z:/marmosets/XROMM_and_RGB_sessions/RGB_videos/validation_2019_04_14and15/'
# x_traj_path = 'Z:/marmosets/XROMM_and_RGB_sessions/XROMM_videos/validation_trajectories/'
# dlcRef = traj_path + '2019_04_15_refFrame.npy'
# xrommRef = x_traj_path + '04_15_axes.csv'

if rigidBody == 'hand':
    mIdx = [0, 3]
    ctIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8] # use if exported marker coordinates were labeled distal-lateral-medial OR immediately after vAv2

elif rigidBody == 'forearm':
    mIdx = [3, 6]
    ctIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8] # use if exported marker coordinates were labeled distal-medial-proximal OR after vAv2

elif rigidBody == 'upperArm':
    mIdx = [6, 9]
    # ctIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8] # use if exported marker coordinates were labeled distal-medial-proximal
    ctIdx = [3, 4, 5, 0, 1, 2, 6, 7, 8] # use if exported marker coordinates were saved immediately after vAv2
elif rigidBody == 'torso':
    mIdx = [9, 12]  
    # ctIdx = [6, 7, 8, 0, 1, 2, 3, 4, 5] # use if exported marker coordinates were labeled back-front-upper
    ctIdx = [3, 4, 5, 6, 7, 8, 0, 1, 2] # use if exported marker coordinates were saved immediately after vAv2
    
# Compute DLC and XROMM basis matrices

# def proj(u, v):   # Projection of u onto v
#     return np.dot(u,v) / np.dot(v,v) * v

# refPoints = np.load(dlcRef)
# refPoints = np.squeeze(refPoints)
# dlc_origin = refPoints[0, :]
# x = -1*(refPoints[1, :] - refPoints[0, :])
# x = x / np.linalg.norm(x)
# y = refPoints[2, :] - refPoints[1, :]
# y = y - proj(y, x)
# y = y / np.linalg.norm(y)
# z = np.cross(x, y)
# z = z / np.linalg.norm(z)
# dlcBasis = np.column_stack((x, y, z))
  
# refPoints = np.loadtxt(xrommRef, delimiter = ',', skiprows = 1)
# refPoints = refPoints[~np.isnan(refPoints[:, 0]), :]
# refPoints = np.reshape(refPoints, (3, 3))
# xromm_origin = refPoints[0, :]
# x = -1*(refPoints[1, :] - refPoints[0, :])
# x = x / np.linalg.norm(x)
# y = refPoints[2, :] - refPoints[1, :]
# y = y - proj(y, x)
# y = y / np.linalg.norm(y)
# z = np.cross(x, y)
# z = z / np.linalg.norm(z)
# xrommBasis = np.column_stack((x, y, z))
    
class rigidBody_coordinates:
    dlc = traj_coordinates[mIdx[0]:mIdx[1], ...]
    # for part in range(dlc.shape[0]):
    #     dlc[part, ...] = dlc[part, ...] - np.repeat(dlc_origin.reshape((3, 1)), dlc.shape[-1], axis = 1)
        # dlc[part, ...] = dlcBasis @ dlc[part, ...]
    #     dlc[part, ...] = np.linalg.inv(xrommBasis) @ dlcBasis @ dlc[part, ...]
    #     dlc[part, ...] = dlc[part, ...] + np.repeat(xromm_origin.reshape((3, 1)), dlc.shape[-1], axis = 1)
    dlc = np.swapaxes(dlc, 0, 1)
    numMarkers = np.size(dlc, 1)
    ct = np.empty((numMarkers, 3))
    ct[0, :] = np.array(ct_coordinates.iloc[0][ctIdx[0:3]])
    ct[1, :] = np.array(ct_coordinates.iloc[0][ctIdx[3:6]])   
    ct[2, :] = np.array(ct_coordinates.iloc[0][ctIdx[6:9]])
    ct = ct.T
#    ct = ct[:, ::-1]
#    prevCoords = ct
#    rotMat = rotx @ roty @ rotz
#    xAvg = xromm_avg
#    dAvg = dlc_avg 
            
#if rigidBody == 'hand':
#    points = rigidBody_coordinates.ct
#    midpoint = np.mean(points[:, 1:], axis = 1)
#    widthVec = points[:, 2] - points[:, 1] / np.linalg.norm(points[:, 2] - points[:, 1])
#    lengthVec = points[:, 0] - midpoint / np.linalg.norm(points[:, 0] - midpoint)
#    normVec = np.cross(lengthVec, widthVec)
#    normVec = normVec / np.linalg.norm(normVec)
#    
#    topVerts = np.vstack((midpoint - 0.5*lengthVec + 0.5*widthVec, midpoint - 0.5*lengthVec - 0.5*widthVec, 
#                          points[:, 2] + 0.5 * lengthVec + 0.5*widthVec, points[:, 2] + 0.5 * lengthVec - 0.5*widthVec)).T
#    topVerts = topVerts[:, (0, 2, 1, 3)]
#    bottomVerts = topVerts - np.tile(np.expand_dims(0.5*normVec, -1), (1, 4))     
   
#
#pt1 = dlcALL.forearm[:, 0]
#pt2 = dlcALL.hand[:, 2]
#pt3 = ctAll.forearm[:, 0]
#pt4 = ctAll.hand[:, 2]  
#distances.forearmDistal_to_handMedial_ct = np.sqrt([np.sum((pt3-pt4)**2)])
#distances.forearmDistal_to_handMedial_dlc = np.sqrt([np.sum((pt1-pt2)**2)])



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

pts = [[0, 0, 1], [1, 2, 2]]

dlc = rigidBody_coordinates.dlc.copy()
dlc[..., :560] = np.nan
dlc[..., 1001:] = np.nan
dlc_tmp = np.empty_like(dlc)
nMarks = rigidBody_coordinates.numMarkers
for frame in range(dlc.shape[-1]):
    src = rigidBody_coordinates.dlc[:, :, frame]
    centroid_src = np.nanmean(src, 1)
    dlc_tmp[..., frame] = src - np.tile(centroid_src, (nMarks, 1)).T
    
dlc = np.nanmean(dlc_tmp, axis = -1)

DLCvectors = np.vstack((dlc[:, pts[0][0]] - dlc[:, pts[1][0]], 
                        dlc[:, pts[0][1]] - dlc[:, pts[1][1]], 
                        dlc[:, pts[0][2]] - dlc[:, pts[1][2]])).T 
DLCdist = np.linalg.norm(DLCvectors, axis = 0)
DLCangles = np.degrees(np.hstack((np.arccos(DLCvectors[:, 0].dot(DLCvectors[:, 1]) / (np.linalg.norm(DLCvectors[:, 0]) * np.linalg.norm(DLCvectors[:, 1]))),
                                  np.pi - np.arccos(DLCvectors[:, 0].dot(DLCvectors[:, 2]) / (np.linalg.norm(DLCvectors[:, 0]) * np.linalg.norm(DLCvectors[:, 2]))),
                                  np.arccos(DLCvectors[:, 1].dot(DLCvectors[:, 2]) / (np.linalg.norm(DLCvectors[:, 1]) * np.linalg.norm(DLCvectors[:, 2]))))))

ct = rigidBody_coordinates.ct.copy()
centroid_ct = np.nanmean(ct, 1)
ct = ct - np.tile(centroid_ct, (nMarks, 1)).T
#ct = ct[:, [0, 2, 1]]

CTvectors = np.vstack((ct[:, pts[0][0]] - ct[:, pts[1][0]], 
                       ct[:, pts[0][1]] - ct[:, pts[1][1]], 
                       ct[:, pts[0][2]] - ct[:, pts[1][2]])).T

CTdist = np.linalg.norm(CTvectors, axis = 0)

CTangles = np.degrees(np.hstack((np.arccos(CTvectors[:, 0].dot(CTvectors[:, 1]) / (np.linalg.norm(CTvectors[:, 0]) * np.linalg.norm(CTvectors[:, 1]))),
                                 np.pi - np.arccos(CTvectors[:, 0].dot(CTvectors[:, 2]) / (np.linalg.norm(CTvectors[:, 0]) * np.linalg.norm(CTvectors[:, 2]))),
                                 np.arccos(CTvectors[:, 1].dot(CTvectors[:, 2]) / (np.linalg.norm(CTvectors[:, 1]) * np.linalg.norm(CTvectors[:, 2]))))))

normCTsep = CTvectors / np.linalg.norm(CTvectors, axis = 0)
diff = CTdist - DLCdist
ratio = CTdist / DLCdist

# ax.scatter(ct[0, 0], ct[1, 0], ct[2, 0], c = 'r', marker = 'o')
# ax.scatter(ct[0, 1], ct[1, 1], ct[2, 1], c = 'r', marker = 'x')
# ax.scatter(ct[0, 2], ct[1, 2], ct[2, 2], c = 'r', marker = '+')

# ax.scatter(dlc[0, 0], dlc[1, 0], dlc[2, 0], c = 'b', marker = 'o')
# ax.scatter(dlc[0, 1], dlc[1, 1], dlc[2, 1], c = 'b', marker = 'x')
# ax.scatter(dlc[0, 2], dlc[1, 2], dlc[2, 2], c = 'b', marker = '+')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()
#ax.scatter(ct[0, :], ct[1, :], ct[2, :], c = 'b', marker = 'o')
#while np.sum(~np.isnan(diff)) > 0:
#    smallDiff = np.nanmin(diff)
#    adjIdx = int(np.where(diff == smallDiff)[0])
#    newDist = CTdist[adjIdx] - smallDiff/2
#    
#    ct[:, pts[1][adjIdx]] = ct[:, pts[1][adjIdx]] + (normCTsep[:, adjIdx] * newDist)
#    ct[:, pts[0][adjIdx]] = ct[:, pts[0][adjIdx]] + (-normCTsep[:, adjIdx] * newDist)   
#    
#    diff[adjIdx] = np.nan
#    
#    CTsep = np.vstack((ct[:, pts[0][0]] - ct[:, pts[1][0]], ct[:, pts[0][1]] - ct[:, pts[1][1]], ct[:, pts[0][2]] - ct[:, pts[1][2]])).T
#    CTdist = np.linalg.norm(CTsep, axis = 0)
#    normCTsep = CTsep / np.linalg.norm(CTsep, axis = 0)
#
#ax.scatter(ct[0, :], ct[1, :], ct[2, :], c = 'r', marker = 'o')
#
#plt.show()

angleOrder = 'o, x, +'

print('DLC angles (blue)')
print(DLCangles)
print('Maya angles (red)')
print(CTangles)
print('DLC distance (blue)')
print(DLCdist)
print('Maya distance (red)')
print(CTdist)

#%%    
#def computePose(frame):
def computePose(frame, rotMat_prev):
    nMarks = rigidBody_coordinates.numMarkers
#    src = np.vstack((rigidBody_coordinates.ct, np.ones((1, nMarks))))
#    dst = np.vstack((rigidBody_coordinates.dlc[:, :, frame], np.ones((1, nMarks))))    

    if direction == 0:    
        dst = rigidBody_coordinates.dlc[:, :, frame]    
        src = rigidBody_coordinates.ct
    elif direction == 1:
        src = rigidBody_coordinates.dlc[:, :, frame]
        dst = rigidBody_coordinates.ct
    
#    dst = dst - np.tile(rigidBody_coordinates.dAvg.reshape((3,1)), (1, 3)) 
#    dst = rigidBody_coordinates.rotMat @ dst
#    dst = dst + np.tile(rigidBody_coordinates.xAvg.reshape((3,1)), (1, 3))

    
    # find Barycentric coordinates (meaning relative to the center of gravity)
    # center of gravity is the mean (x, y, z) position of the markers
    centroid_src = np.nanmean(src, 1)
    centroid_dst = np.nanmean(dst, 1)
    src = src - np.tile(centroid_src, (nMarks, 1)).T
    dst = dst - np.tile(centroid_dst, (nMarks, 1)).T

    if direction == 0:    
        cov = src @ dst.T
        U, S, Vt = np.linalg.svd(cov)
        V = Vt.T
        sign = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(U) * np.linalg.det(V)]])
#        rotMat = U @ sign @ V.T
        rotMat = V @ sign @ U.T
    
    elif direction == 1:
    #    cov = dst @ src.T
        cov = src @ dst.T
        U, S, Vt = np.linalg.svd(cov)
        V = Vt.T
        sign = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(U) * np.linalg.det(V)]])
        rotMat = U @ sign @ Vt

#    if np.sum(rotMat != 0):
#        rotVec_prev = rotMat_prev.flatten()
#        rotVec = rotMat.flatten()
#        dot_over_mag = np.dot(rotVec_prev, rotVec) / (np.sqrt(np.dot(rotVec, rotVec)) * np.sqrt(np.dot(rotVec_prev, rotVec_prev)))    
#        angle = np.arccos(dot_over_mag)
#        if (180 / np.pi * angle > 150):
#            ang = np.sqrt(np.dot(rotVec, rotVec))
#            rotMat = -rotMat / ang * (2*np.pi - ang)

    if direction == 0:    
        trans = -rotMat @ centroid_src.reshape((3,1)) + centroid_dst.reshape((3,1))
    elif direction == 1:
        trans = -rotMat @ centroid_dst.reshape((3,1)) + centroid_src.reshape((3,1))
    
#    flatRT = np.hstack((rotMat[0, :], 0, rotMat[1, :], 0, rotMat[2, :], 0, trans.reshape((3)), 1))
    flatRT = np.hstack((rotMat[:, 0], 0, rotMat[:, 1], 0, rotMat[:, 2], 0, trans.reshape((3)), 1))
    
    return flatRT, rotMat, trans

rigid_body_transformations = np.empty((rigidBody_coordinates.dlc.shape[-1], 16))
dstPrime = np.empty((3, 3, rigidBody_coordinates.dlc.shape[-1]))
rmse = np.empty((rigidBody_coordinates.dlc.shape[-1]))
normVec = np.empty((rigidBody_coordinates.dlc.shape[-1], 3))
realNormVec = np.empty_like(normVec)
allTopVerts = np.empty((3, 4, rigidBody_coordinates.dlc.shape[-1]))
allBottomVerts = np.empty_like(allTopVerts)
rotMat = np.zeros((3, 3))
for frame in range(traj_coordinates.shape[-1]):
    if np.isnan(np.nanmean(traj_coordinates[..., frame])):
        rigid_body_transformations[frame, :] = np.nan
        dstPrime[:, :, frame] = np.nan
        rmse[frame] = np.nan
        normVec[frame, :] = np.nan
    else:
        rigid_body_transformations[frame, :], rotMat, trans = computePose(frame, rotMat) #computePose(frame)
        dstPrime[:, :, frame] = rotMat @ rigidBody_coordinates.ct + np.tile(trans, (1, 3))
        dstPrime[:, :, frame] = dstPrime[:, :, frame]
        
#        allTopVerts[..., frame] = rotMat @ topVerts + np.tile(trans, (1, 4))
#        allBottomVerts[..., frame] = rotMat @ bottomVerts + np.tile(trans, (1, 4))
        
#        dstPrime[:, :, frame] = rotMat @ rigidBody_coordinates.dlc[...,frame] + np.tile(trans, (1, 3))
                
        # translate true DLC and rigid_transform coords to origin
#        transformCenter = np.mean(dstPrime[:, :, frame], axis = 1)
#        dstPrime[..., frame] = dstPrime[..., frame] - np.tile(transformCenter, (3, 1)).T
#        dlc = rigidBody_coordinates.dlc.copy()[:, :, frame]
#        realCenter = np.mean(dlc, axis = 1)
#        dlc = dlc - np.tile(realCenter, (3, 1)).T
#
        # compute normal vectors to planes of both coordinate sets
        normVec[frame, :] = np.cross(dstPrime[:, 0, frame] - dstPrime[:, 1, frame], dstPrime[:, 0, frame] - dstPrime[:, 2, frame])
        normVec[frame, :] = normVec[frame, :] / np.linalg.norm(normVec[frame, :])
        realNormVec[frame, :] = np.cross(dlc[:, 0] - dlc[:, 1], dlc[:, 0] - dlc[:, 2])
        realNormVec[frame, :] = -1 * realNormVec[frame, :] / np.linalg.norm(realNormVec[frame, :])
#        
#        # rotate normal vector (and coordinates) of rigid transformation to match DLC true data
#        axis = np.cross(normVec[frame, :], realNormVec[frame, :])
#        sin = np.linalg.norm(axis)
#        cos = np.dot(normVec[frame, :], realNormVec[frame, :])
#        skewMat = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
#        rotation = np.identity(3) + skewMat + skewMat @ skewMat * (1 - cos)/(sin * sin)
#        dstPrime[..., frame] = rotation @ dstPrime[..., frame]
#
##        print((rotation @ normVec[frame, :], realNormVec[frame, :]))
#        
#        # rotate rigid transform coordinates about normal vector to match dlc coordinates 
#        vec1 = dstPrime[:, 0, frame]
#        vec2 = dlc[:, 0]
#        normVec[frame, :] = rotation @ normVec[frame, :]       
#                
#        cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
#        sin = np.linalg.norm(np.cross(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
#        theta = np.arccos(cos)
#        
#        qPt1 = Quaternion(np.insert(vec1, 0, 0))
#        qPt2 = Quaternion(np.insert(dstPrime[:, 1, frame], 0, 0))
#        qPt3 = Quaternion(np.insert(dstPrime[:, 2, frame], 0, 0))
#        q2 = Quaternion(np.hstack((np.array([theta/2]), normVec[frame, :] * np.sin(theta/2))))
#        q3 = q2 * qPt1 * q2.inverse
#        rotPoint1 = q3.elements[1:]
#        q3 = q2 * qPt2 * q2.inverse
#        rotPoint2 = q3.elements[1:]
#        q3 = q2 * qPt3 * q2.inverse
#        rotPoint3 = q3.elements[1:]        
#
##        rotation = (1 - cos) * np.dot(vec1, normVec[frame, :]) * normVec[frame, :] + cos * vec1 + sin * np.cross(normVec[frame, :], vec1)
#        dstPrime[..., frame] = np.vstack((rotPoint1, rotPoint2, rotPoint3)).T + np.tile(transformCenter, (3, 1)).T
##        dstPrime[..., frame] = rotation @ dstPrime[..., frame] + np.tile(transformCenter, (3, 1)).T        
#        
#        normVec[frame, :] = np.cross(dstPrime[:, 0, frame] - dstPrime[:, 1, frame], dstPrime[:, 0, frame] - dstPrime[:, 2, frame])
#        normVec[frame, :] = normVec[frame, :] / np.linalg.norm(normVec[frame, :])
        
        err = dstPrime[:, :, frame] - rigidBody_coordinates.dlc[:, :, frame] 
        # err = dstPrime[:, :, frame] - rigidBody_coordinates.ct 
        rmse[frame] = np.sqrt(np.mean(err.flatten() * err.flatten()))

b, a = signal.butter(4, 0.1)
tmp =rigid_body_transformations[~np.isnan(rigid_body_transformations[:, 0]), :]
rigid_body_transformations[~np.isnan(rigid_body_transformations[:, 0]), :] = signal.filtfilt(b, a, tmp, axis = 0)

totalRMSE = np.nanmean(rmse)
print(totalRMSE)

rigid_body_transformations[:540, :]  = np.nan
rigid_body_transformations[1075:, :] = np.nan
rigid_body_transformations = pd.DataFrame(rigid_body_transformations)
rigid_body_transformations.columns = [rigidBody + '_R11', rigidBody + '_R12', 
                                     rigidBody + '_R13', rigidBody + '_R01', 
                                     rigidBody + '_R21', rigidBody + '_R22', 
                                     rigidBody + '_R23', rigidBody + '_R02', 
                                     rigidBody + '_R31', rigidBody + '_R32', 
                                     rigidBody + '_R33', rigidBody + '_R03', 
                                     rigidBody + '_TX', rigidBody + '_TY', 
                                     rigidBody + '_TZ', rigidBody + '_1']

rigid_body_transformations.to_csv('Z:/marmosets/XROMM_and_RGB_sessions/new_skin_and_bones_objects_pat/rigid_body_transformations/rigid_body_transformations_%s.csv' % rigidBody, index=False, na_rep='NaN')

##### plotting animated 3D movement of rigid body #######

def animate(frame):
#    print(frame)
    idx = [frame, frame+1]
    centers = np.mean(dstPrime[:, :, idx[0]:idx[1]], axis = 1)
    scat_data = dstPrime[:, :, idx[0]:idx[1]]
    scat_data = np.swapaxes(scat_data, 0, 2).reshape((centers.shape[-1] * dstPrime.shape[1], 3))
    scat._offsets3d = ([scat_data[0, 0]], [scat_data[0, 1]], [scat_data[0, 2]])
    scat2._offsets3d = ([scat_data[1, 0]], [scat_data[1, 1]], [scat_data[1, 2]])
    scat3._offsets3d = ([scat_data[2, 0]], [scat_data[2, 1]], [scat_data[2, 2]])

    
    real_data = rigidBody_coordinates.dlc[:, :, idx[0]:idx[1]]
    real_data = np.swapaxes(real_data, 0, 2).reshape((centers.shape[-1] * dstPrime.shape[1], 3))
    realScat._offsets3d = ([real_data[0, 0]], [real_data[0, 1]], [real_data[0, 2]])
    realScat2._offsets3d = ([real_data[1, 0]], [real_data[1, 1]], [real_data[1, 2]])
    realScat3._offsets3d = ([real_data[2, 0]], [real_data[2, 1]], [real_data[2, 2]])
    
    arrow_data = np.hstack((centers[:, -1], centers[:, -1] + normVec[idx[1], :]))
    arrow_segs = [[[x,y,z], [u,v,w]] for x,y,z,u,v,w in arrow_data.reshape((1, len(arrow_data)))]
    arrow.set_segments(arrow_segs)
    
    realCenters = np.mean(rigidBody_coordinates.dlc[:, :, idx[0]:idx[1]], axis = 1)
    realArrow_data = np.hstack((realCenters[:, -1], realCenters[:, -1] + realNormVec[idx[1], :]))
    realArrow_segs = [[[x,y,z],[u,v,w]] for x,y,z,u,v,w in realArrow_data.reshape((1, len(arrow_data)))]
    realArrow.set_segments(realArrow_segs)
    
    # topVerts = allTopVerts[..., frame]
    # bottomVerts = allBottomVerts[..., frame]
    # leftVerts = np.vstack((topVerts[:, 0], topVerts[:, 2], bottomVerts[:, 0], bottomVerts[:, 2])).T    
    # rightVerts = np.vstack((topVerts[:, 1], topVerts[:, 3], bottomVerts[:, 1], bottomVerts[:, 3])).T    
    # frontVerts = np.hstack((topVerts[:, 2:], bottomVerts[:, 2:]))    
    # backVerts = np.hstack((topVerts[:, :2], bottomVerts[:, :2]))      
    # topVerts =    [[tuple(v) for v in np.round(topVerts.T).astype(int)]]
    # bottomVerts = [[tuple(v) for v in np.round(bottomVerts.T).astype(int)]]
    # leftVerts =   [[tuple(v) for v in np.round(leftVerts.T).astype(int)]]
    # rightVerts =  [[tuple(v) for v in np.round(rightVerts.T).astype(int)]]
    # frontVerts =  [[tuple(v) for v in np.round(frontVerts.T).astype(int)]]
    # backVerts =   [[tuple(v) for v in np.round(backVerts.T).astype(int)]]

    # top.set_verts(topVerts)
    # bottom.set_verts(bottomVerts)
    # left.set_verts(leftVerts)
    # right.set_verts(rightVerts)
    # front.set_verts(frontVerts)
    # back.set_verts(backVerts)
    
    # verts = [[(0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 1, 1)]]

    # print((np.nanmean(real_data[-1, :]) - np.nanmean(scat_data[-1, :]), np.degrees(np.arccos(np.dot(normVec[idx[1], :], realNormVec[idx[1], :])))))


#%%
# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter([], [], [], c = 'r', marker = 'o')
scat2 = ax.scatter([], [], [], c = 'r', marker = 'x')
scat3 = ax.scatter([], [], [], c = 'r', marker = '+')
realScat = ax.scatter([], [], [], c = 'b', marker = 'o')
realScat2 = ax.scatter([], [], [], c = 'b', marker = 'x')
realScat3 = ax.scatter([], [], [], c = 'b', marker = '+')
arrow = ax.quiver([], [], [], [], [], [])
realArrow = ax.quiver([], [], [], [], [], [])

arrow.set_color('r')

# x_range = np.arange(0, 1+1/10, 1/10)
# y_range = np.arange(0, 1+1/10, 1/10)
# xx, yy = np.meshgrid(x_range, y_range)
# zz = np.ones_like(xx) * 2
# top = ax.plot_surface(xx, yy, zz, color="b", alpha=0.8)
# zz = np.ones_like(xx) * 0
# bottom = ax.plot_surface(xx, yy, zz, color="r", alpha=0.2)

# x_range = np.arange(0, 1+1/10, 1/10)
# z_range = np.arange(0, 2+1/10, 1/5)
# xx, zz = np.meshgrid(x_range, z_range)
# yy = np.ones_like(xx) * 1
# left = ax.plot_surface(xx, yy, zz, color="r", alpha=0.2)
# yy = np.ones_like(xx) * 0
# right = ax.plot_surface(xx, yy, zz, color="r", alpha=0.2)

# z_range = np.arange(0, 2+1/10, 1/5)
# y_range = np.arange(0, 1+1/10, 1/10)
# yy, zz = np.meshgrid(y_range, z_range)
# xx = np.ones_like(xx) * 1
# front = ax.plot_surface(xx, yy, zz, color="r", alpha=0.2)
# xx = np.ones_like(xx) * 0
# back = ax.plot_surface(xx, yy, zz, color="r", alpha=0.2)

################################################3

minDims = np.nanmin(np.nanmin(dstPrime, axis = -1), axis = -1)
maxDims = np.nanmax(np.nanmax(dstPrime, axis = -1), axis = -1)

#minDims = np.nanmin(np.nanmin(rigidBody_coordinates.dlc, axis = -1), axis = -1)
#maxDims = np.nanmax(np.nanmax(rigidBody_coordinates.dlc, axis = -1), axis = -1)

# Setting the axes properties
ax.set_xlim3d(minDims[0], maxDims[0])
ax.set_xlabel('X')

ax.set_ylim3d([minDims[1], maxDims[1]])
ax.set_ylabel('Y')

ax.set_zlim3d([minDims[2], maxDims[2]])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, animate, frames = np.arange(560, 1001), interval = 50, repeat = True)
plt.show()

# line_ani.save(r'C:\Users\daltonm\Documents\Lab_Files\rigid_body_computation\hand_rigidBody.gif')