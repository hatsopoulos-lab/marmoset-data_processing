# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:27:30 2021

@author: Dalton
"""

import sys
import glob
import numpy as np
import shutil
import os

filePattern = sys.argv[1]
drop_record_path = sys.argv[2]
fps = sys.argv[3]

period_ns = 1/int(fps) * 1e9

frames = sorted(glob.glob(filePattern))

event = frames[0].split('event_')[1][:3] 
subject_date_exp = os.path.basename(frames[0]).split('_session')[0]

timestamps = []
frameNums = []
frameIdx = []
for fr, file in enumerate(frames):
    timestamps.append(int(file.split('currentTime_')[1][:-12]))
    frameNums.append(int(file.split('frame_')[1][:7]))
    frameIdx.append(fr)

frameDiffs = np.diff(frameNums)
dropFrames = np.where(frameDiffs > 1)[0]

# check if last frame was dropped by comparing to other camera folders
bases = [filePattern.split('jpg_cam')[0]]

lastFrameNums = []
dropRecord = []
for base in bases:
    cam_folders = glob.glob(os.path.join(base, 'jpg_cam*'))
    for f in cam_folders:
        lastFrameNums.append(int(sorted(glob.glob(os.path.join(f, '%s*event_%s*' %(subject_date_exp, event))))[-1].split('frame_')[1][:7]))            

# append values to frameNums and dropFrames so that the dropped frame(s) at the
# end will be replaced
if frameNums[-1] < max(lastFrameNums):
    frameDiffs = np.append(frameDiffs, max(lastFrameNums) - frameNums[-1] + 1)
    dropFrames = np.append(dropFrames, len(frameNums)-1)
    dropRecord.append(max(lastFrameNums))


# timeDiffs = np.diff(timestamps)
# drop_locs = np.where(timeDiffs > 1.1*period_ns)[0]
if len(dropFrames) > 0:
    print('\n\n\n\n\n\n got some wrong frame nums!! \n\n\n\n\n\n\n\n\n')
    
    sessDir, cam = os.path.split(os.path.dirname(frames[0]))
    cam = cam.split('jpg_')[1]
    dateDir, session = os.path.split(sessDir)  
    f = open(os.path.join(drop_record_path, subject_date_exp+'_'+session+'_'+cam+'_'+'event_'+event+'_droppedFrames.txt'), 'w')
    
    for dFr in dropFrames:
        lastGoodFrame, lastGoodTime, lastGoodFile = frameNums[dFr], timestamps[dFr], frames[dFr]        
        print('\n\n')
        for copyNum in range(1, frameDiffs[dFr]):
            newFrame = lastGoodFrame + copyNum
            newFile = lastGoodFile.replace('frame_'+str(lastGoodFrame).zfill(7), 'frame_'+str(newFrame).zfill(7))
            newFile = newFile.replace('currentTime_'+str(lastGoodTime), 'currentTime_'+str(lastGoodTime + period_ns*copyNum))

            shutil.copyfile(lastGoodFile, newFile)
            
            dropRecord.append(newFrame)
    
    dropRecord = sorted(dropRecord)

    for fr in dropRecord:
        f.write(str(fr) + ',') 
    
    f.close()
            

        
    



                  
 

