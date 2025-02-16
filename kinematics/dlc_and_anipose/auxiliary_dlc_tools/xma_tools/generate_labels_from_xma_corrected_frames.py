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
need_to_correct_number_of_labeled_points = False 
#############################################################

class params:
    label_to_focus = 'l-wrist'
    thresh = 35
    medfilt_length= 81
    
event_pattern = re.compile('e\d{3}')
sess_pattern  = re.compile('_s\d{1}_')
long_event_pattern = re.compile('event\d{3}')
long_sess_pattern  = re.compile('session\d{1}')
cam_pattern   = re.compile('cams_\d{1}_and_\d{1}')

# corrections_path = r'C:\Users\Dalton\Documents\lab_files\dlc_corrected_files'
# config = r'C:\Users\Dalton\Documents\lab_files\dlc_local\simple_joints_model-Dalton-2021-04-08\config.yaml'
corrections_path = '/media/marms/DATA/dlc_corrected_files'
config = '/home/marms/Documents/dlc_local/simple_joints_model-Dalton-2021-04-08/config.yaml'
videos = sorted(glob.glob(os.path.join(corrections_path, 'videos', '*')))
traj_paths_original  = sorted(glob.glob(os.path.join(corrections_path, 'original_trajectories' , '*')))
traj_paths_corrected = sorted(glob.glob(os.path.join(corrections_path, 'corrected_trajectories', '*')))

def xma_to_h5(path_dict, scorer):
    """ adapted from xma_to_dlc() in XROMM_DLC_tools by J.D. Laurence-Chasen to simply export corrected labels for given event
        as .h5 and .csv to put back into DLC postprocessing pipline
    """
    
    # grab corrected data
    corrections = sorted(glob.glob(os.path.join(path_dict['xma_to_dlc'], '*')))
    corrections = [Path(cp) for cp in corrections]
    
    videos               = []
    traj_paths_original  = []
    traj_paths_corrected = []
    for xma_corr in corrections:
        xma_stem = xma_corr.stem
        cam_text = re.search(cam_pattern, xma_stem)[0]
        cams = cam_text.split('cams_')[-1].split('_and_')
        cams = [int(cam) for cam in cams]
        
        # df=pd.read_csv(xma_corr, sep=',',header=None)
        # # read pointnames from header row
        # pointnames = df.loc[0,::4].astype(str).str[:-7].tolist()
        # df_tmp = df.loc[1:,].reset_index(drop=True) # remove header row
        
        # reload with header so data has dtype float, then grab pointnames from header
        df=pd.read_csv(xma_corr, sep=',',header=0)
        pointnames = df.columns[::4].astype(str).str[:-7].tolist()
        for nan_txt in ['', ' NaN', ' NaN ', 'NaN ']:
            df.replace(nan_txt, np.nan, inplace=True)
        df.apply(pd.to_numeric)

        
# videos = [Path(path_dict['corrections_vids']) / Path(cp.stem.sp)for vp in ]
    
    
    # convert corrected data to dataframes and save in corrected_trajectories
    
    # return video paths, corrected_trajectory paths, and original trajectory paths

        for cIdx, cam in enumerate(cams):
            
            # identify video paths and original trajectories corresponding to corrected data
            base = xma_stem.replace(cam_text, 'cam%d' % cam).split(scorer)[0]
            vid_path = os.path.join(path_dict['video_files'], base + '.avi')
            orig_traj_path = os.path.join(path_dict['corrections_orig'], base + scorer + '.h5')
            corr_traj_path = os.path.join(path_dict['corrections_corr'], base + scorer + '.h5')
            videos.append(vid_path)
            traj_paths_original.append(orig_traj_path)
            traj_paths_corrected.append(corr_traj_path)
            
            # Load original trajectory to get original index and column info
            orig_traj = pd.read_hdf(orig_traj_path)
            corr_traj = orig_traj.copy()
            
            # extract 2D points data
            xpos = df.iloc[:, 0+cIdx*2::4]
            ypos = df.iloc[:, 1+cIdx*2::4]
            
            # place data in correct columns in corr_traj
            pdIdx = pd.IndexSlice
            for i, bodypart in enumerate(pointnames):
                x_header = [col for col in xpos.columns if bodypart in col][0]
                y_header = [col for col in ypos.columns if bodypart in col][0]

                corr_traj.loc[:, pdIdx[:, bodypart, 'x']] = xpos.loc[:, x_header]
                corr_traj.loc[:, pdIdx[:, bodypart, 'y']] = ypos.loc[:, y_header]
            
            corr_traj.to_hdf(corr_traj_path, key="df_with_missing", mode="w")

    return videos, traj_paths_original, traj_paths_corrected 

def identify_corrected_frames(original_path, corrected_path, label_to_focus):
    
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
    
    identified_corrections = np.array(traj_orig.loc[:, pdIdx[:, label_to_focus, 'x']]).squeeze()
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

def generate_labels(image_names, traj_corrections, config, label_dir, indexlength):
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
    
    image_names = [os.fspath(Path(*label_dir.parts[-2:]) /  ("img" + str(imageNum).zfill(indexlength) + ".png")) 
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

def process_corrected_trajectories(videos, traj_paths_original, traj_paths_corrected, label_dirs, label_to_focus):
    for video, orig_traj_path, corr_traj_path, label_dir in zip(videos, 
                                                                traj_paths_original, 
                                                                traj_paths_corrected,
                                                                label_dirs):
        
        cap = VideoReader(video)
        nframes = len(cap)
        indexlength = int(np.ceil(np.log10(nframes)))
        fname = Path(video)
        output_path = Path(config).parents[0] / "labeled-data" / fname.stem
        
        frames2pick, traj_corrections = identify_corrected_frames(orig_traj_path, corr_traj_path, label_to_focus)
        
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
        generate_labels(image_names, traj_corrections, config, label_dir, indexlength)
    
if __name__ == "__main__":
    
    args = {'proj_dir': '/project/nicho/data/marmosets/kinematics_videos/moths/TYJL/2021_02_11',
            'config'  : '/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26/config.yaml'}
    
    label_to_focus = 'l-wrist'
      
    basedir = args['proj_dir']
    
    path_dict = {'proj_dir'         : basedir,
                 'corrections_orig' : os.path.join(basedir, 'xmalab_corrections', 'original_trajectories'),
                 'corrections_corr' : os.path.join(basedir, 'xmalab_corrections', 'corrected_trajectories'),
                 'video_files'      : os.path.join(basedir, 'avi_videos'),
                 'xma_to_dlc'       : os.path.join(basedir, 'xmalab_corrections', 'xma_to_dlc')}

    with open(os.path.join(basedir, 'scorer_info.txt')) as f:
        scorer = f.readlines()[0]
        
    scorer = scorer.split('filtered')[-1]
    scorer = scorer.split('_meta')[0]
    
    videos, traj_paths_original, traj_paths_corrected = xma_to_h5(path_dict, scorer)
    
    label_dirs = add_videos_to_project(args['config'], videos)
    # label_dirs = [Path('/home/marms/Documents/dlc_local/simple_joints_model-Dalton-2021-04-08/TYJL_2021_02_11_foraging_session1_event049_cam1_filtered'),
    #               Path('/home/marms/Documents/dlc_local/simple_joints_model-Dalton-2021-04-08/TYJL_2021_02_11_foraging_session1_event049_cam2_filtered')]
    process_corrected_trajectories(videos, traj_paths_original, traj_paths_corrected, label_dirs, label_to_focus)
    