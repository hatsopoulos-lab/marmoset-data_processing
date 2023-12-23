#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 06:26:01 2023

@author: daltonm
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re
import cv2

date = '2023_08_04'
exp = 'moth_free'
marms = 'JLTY'
cam = 3
all_cams = [1, 2, 3, 4]
session = 1

frame_time_pattern = re.compile('[0-9]{12,14}')
frame_num_pattern  = re.compile('[0-9]{7}')

kinematics_jpgs_path = Path('/scratch/midway3/daltonm/kinematics_jpgs') / exp / marms / date / f'session{session}' / f'jpg_cam{cam}'

other_cams_jpgs_paths = [kinematics_jpgs_path.parent / f'jpg_cam{otherCam}' for otherCam in all_cams if otherCam != cam]

normal_video_frame_counts = [len(list(jpg_path.glob('*.jpg'))) for jpg_path in other_cams_jpgs_paths] 
normal_video_frame_count  = int(np.mean(normal_video_frame_counts)) 

short_video_frame_list = sorted(list(kinematics_jpgs_path.glob('*.jpg')))
short_video_frame_count = len(short_video_frame_list)

short_video_frame_times = [int(re.findall(frame_time_pattern, frame.stem)[0]) for frame in short_video_frame_list]
short_video_frame_nums  = [int(re.findall(frame_num_pattern, frame.stem)[0]) for frame in short_video_frame_list]

frame_num_diff  = 1
frame_time_diff = int(np.diff(short_video_frame_times).mean().round())

template_img = cv2.imread(str(short_video_frame_list[0])) 
img = np.zeros_like(template_img)
for frame_idx in range(short_video_frame_count, normal_video_frame_count): 
    new_frame_time = str(short_video_frame_times[frame_idx-1] + frame_time_diff)
    new_frame_num  = str(short_video_frame_nums [frame_idx-1] + frame_num_diff).zfill(7)
    new_frame_path = re.sub(frame_num_pattern , new_frame_num , str(short_video_frame_list[frame_idx-1]))
    new_frame_path = re.sub(frame_time_pattern, new_frame_time, new_frame_path)
    short_video_frame_list.append(Path(new_frame_path))
    short_video_frame_times.append(int(new_frame_time))
    short_video_frame_nums.append( int(new_frame_num))
    # print(new_frame_path)
    cv2.imwrite(new_frame_path, img)