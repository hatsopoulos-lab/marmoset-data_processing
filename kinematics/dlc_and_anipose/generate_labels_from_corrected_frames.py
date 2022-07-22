# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:37:32 2022

@author: Dalton
"""

import glob
import os
import pandas as pd
import cv2
import shutil
import numpy as np
import matplotlib
import re
import matplotlib.pyplot as plt
from deeplabcut.utils.auxfun_videos import VideoReader
from deeplabcut.utils import auxiliaryfunctions
from pathlib import Path
from skimage import io
from skimage.util import img_as_ubyte
from scipy.signal import medfilt

########## NOTE #############################################
# if this is True, modify the specifics of the corrected in indentify_corrected_frames --> if need_to_correct_number_of_labeled_points:
need_to_correct_number_of_labeled_points = True 
#############################################################

class params:
    label_to_focus = 'hand'
    thresh = 35
    medfilt_length= 81

# corrections_path = r'C:\Users\Dalton\Documents\lab_files\dlc_corrected_files'
# config = r'C:\Users\Dalton\Documents\lab_files\dlc_local\simple_joints_model-Dalton-2021-04-08\config.yaml'
corrections_path = '/media/marms/DATA/dlc_corrected_files'
config = '/home/marms/Documents/dlc_local/simple_joints_model-Dalton-2021-04-08/config.yaml'
videos = sorted(glob.glob(os.path.join(corrections_path, 'videos', '*')))
traj_paths_original  = sorted(glob.glob(os.path.join(corrections_path, 'original_trajectories' , '*')))
traj_paths_corrected = sorted(glob.glob(os.path.join(corrections_path, 'corrected_trajectories', '*')))

def identify_corrected_frames(original_path, corrected_path):
    
    traj_orig = pd.read_hdf(original_path)
    traj_corr = pd.read_hdf(corrected_path)

    if need_to_correct_number_of_labeled_points:
        traj_orig = traj_orig.iloc[:, :-9]
        
    pdIdx = pd.IndexSlice
    label_distance = np.add(np.square(np.array(traj_corr.loc[:, pdIdx[:, :, 'x']]) - np.array(traj_orig.loc[:, pdIdx[:, :, 'x']])),
                            np.square(np.array(traj_corr.loc[:, pdIdx[:, :, 'y']]) - np.array(traj_orig.loc[:, pdIdx[:, :, 'y']])))
    label_distance = np.sqrt(label_distance)
    
    label_names = []
    for scorer, part, coord in traj_orig.columns:
        if part not in label_names:
            label_names.append(part)
    label_focus_idx = [idx for idx, name in enumerate(label_names) if name == params.label_to_focus][0]
    
    label_distance = label_distance[:, label_focus_idx]
    
    label_distance_filtered = medfilt(label_distance, params.medfilt_length)
    label_distance_from_median = label_distance - label_distance_filtered
    
    identified_corrections = np.array(traj_orig.loc[:, pdIdx[:, 'hand', 'x']]).squeeze()
    identified_corrections[label_distance_from_median < params.thresh] = np.nan
        
    # fig, ax = plt.subplots()
    # ax.plot(traj_corr.loc[:, pdIdx[:, 'hand', 'x']])
    # ax.plot(traj_orig.loc[:, pdIdx[:, 'hand', 'x']])
    # ax.plot(label_distance_from_median)    
    # ax.hlines(params.thresh, 0, len(label_distance))
    # ax.plot(identified_corrections, '-o')
    # plt.show()
    
    # fig, ax = plt.subplots()
    # ax.plot(traj_corr.loc[:, pdIdx[:, 'hand', 'x']])
    # ax.plot(traj_orig.loc[:, pdIdx[:, 'hand', 'x']])
    # ax.plot(label_distance)    
    # ax.plot(label_distance_filtered)
    # plt.show()
    
    frames2pick = np.where(label_distance_from_median > params.thresh)[0]
    
    return frames2pick, traj_corr.iloc[frames2pick, :]

def add_videos_to_project(config, videos):
    cfg = auxiliaryfunctions.read_config(config)
    
    video_path = Path(config).parents[0] / "videos"
    data_path = Path(config).parents[0] / "labeled-data"
    videos = [Path(vp) for vp in videos]

    dirs = [data_path / Path(i.stem) for i in videos]

    for p in dirs:
        p.mkdir(parents=True, exist_ok=True)

    destinations = [video_path.joinpath(vp.name) for vp in videos]
    for src, dst in zip(videos, destinations):
        if dst.exists():
            pass
        else:
            print("Copying the videos")
            shutil.copy(os.fspath(src), os.fspath(dst))

    for idx, video in enumerate(destinations):
        try:
            video_path = str(Path.resolve(Path(video)))
        except:
            video_path = os.readlink(video)

        vid = VideoReader(video_path)
        c = vid.get_bbox()
        params = {video_path: {"crop": ", ".join(map(str, c))}}
        if "video_sets_original" not in cfg:
            cfg["video_sets"].update(params)
        else:
            cfg["video_sets_original"].update(params)

    auxiliaryfunctions.write_config(config, cfg)
    
    return dirs

def generate_labels(image_names, traj_corrections, config, label_dir):
    # grab existing label set to ensure generated labels take same format
    
    existing_label_dirs = glob.glob(os.path.join(os.path.split(label_dir)[0], '*')) 
    
    corrected_event = str(label_dir).split('event')[1][:3]
    cam = 'cam' + str(label_dir).split('cam')[1][0]
    pattern = re.compile(r'\d{4}_\d{2}_\d{2}')
    date = re.findall(pattern, os.path.basename(label_dir))[0]
    
    # prioritze grabbing existing labels from the same date and nearest preceding event
    # so that the origin and axes points are most likely to be correct
    same_date_label_dirs = [folder for folder in existing_label_dirs 
                              if cam in folder 
                              and date in folder 
                              and corrected_event not in folder] 
    
    difference_event = np.array([int(corrected_event) - int(os.path.basename(folder).split('event')[1][:3]) 
                                 for folder in same_date_label_dirs])
    difference_event[difference_event < 0] = difference_event[difference_event < 0] * -100
    same_date_label_dirs = [folder for _, folder in sorted(zip(difference_event, same_date_label_dirs), 
                                                           key=lambda pair: pair[0])]
    other_date_label_dirs = [folder for folder in existing_label_dirs
                             if cam in folder
                             and date not in folder]
    existing_label_dirs = same_date_label_dirs + other_date_label_dirs
    
    # run through label folders until we find a usable set of labels
    for labDir in existing_label_dirs:
        try:
            datapath = glob.glob(os.path.join(labDir, 'CollectedData*.h5'))[0]
            example_label_coords = pd.read_hdf(datapath)
            break
        except:
            continue
    
    digits = 4
    image_names = [os.fspath(Path(*label_dir.parts[-2:]) /  ("img" + str(imageNum).zfill(digits) + ".png")) 
                   for imageNum in traj_corrections.index]
    new_labels = pd.DataFrame(data    = np.full((traj_corrections.shape[0], example_label_coords.shape[1]), np.nan),
                              index   = image_names,
                              columns = example_label_coords.columns) 
    
    for correct_col in traj_corrections.columns:
        for new_col in new_labels.columns:
            if correct_col[1] == new_col[1] and correct_col[2] == new_col[2]: 
                new_labels[new_col] = np.array(traj_corrections[correct_col])
    
    pdIdx = pd.IndexSlice
    if all(new_labels.loc[:, pdIdx[:, 'origin', 'x']] == np.nan):
        origin_and_axis_coords = np.expand_dims(np.array(example_label_coords.loc[:, pdIdx[:, ['origin', 'x', 'y'], :]])[0, :], axis=0)
        new_labels.loc[:, pdIdx[:, ['origin', 'x', 'y'], :]] = np.tile(origin_and_axis_coords, (new_labels.shape[0], 1)) 
    
    # just need to save this to the new CollectedData file and check the labels fill in on the marmoset DLC computer
    scorer = new_labels.columns[0][0]
    new_labels.to_csv(os.path.join(label_dir, "CollectedData_" + scorer + ".csv"))
    new_labels.to_hdf(os.path.join(label_dir, "CollectedData_" + scorer + ".h5"), "df_with_missing", format="table", mode="w")
    
    return

def process_corrected_trajectories(videos, traj_paths_original, traj_paths_corrected, label_dirs):
    for video, orig_traj_path, corr_traj_path, label_dir in zip(videos, 
                                                                traj_paths_original, 
                                                                traj_paths_corrected,
                                                                label_dirs):
        
        cap = VideoReader(video)
        nframes = len(cap)
        indexlength = int(np.ceil(np.log10(nframes)))
        fname = Path(video)
        output_path = Path(config).parents[0] / "labeled-data" / fname.stem
        
        frames2pick, traj_corrections = identify_corrected_frames(orig_traj_path, corr_traj_path)
        
        print('\nextracting images from ' + Path(video).stem)
        
        output_path = (Path(config).parents[0] / "labeled-data" / Path(video).stem)
        image_names = []
        for index in frames2pick:
            cap.set_to_frame(index)  # extract a particular frame
            frame = cap.read_frame()
            if frame is not None:
                image = img_as_ubyte(frame)
                img_name = (str(output_path) + "/img" + str(index).zfill(indexlength) + ".png")
                io.imsave(img_name, image)
                image_names.append(img_name)
        cap.close()
        image_names = []
        generate_labels(image_names, traj_corrections, config, label_dir)
    
if __name__ == "__main__":
    label_dirs = add_videos_to_project(config, videos)
    # label_dirs = [Path('/home/marms/Documents/dlc_local/simple_joints_model-Dalton-2021-04-08/TYJL_2021_02_11_foraging_session1_event049_cam1_filtered'),
    #               Path('/home/marms/Documents/dlc_local/simple_joints_model-Dalton-2021-04-08/TYJL_2021_02_11_foraging_session1_event049_cam2_filtered')]
    process_corrected_trajectories(videos, traj_paths_original, traj_paths_corrected, label_dirs)
    