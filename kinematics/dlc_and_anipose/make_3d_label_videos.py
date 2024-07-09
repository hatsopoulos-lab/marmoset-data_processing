#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:19:50 2024

@author: daltonm
@derived_from: https://github.com/lambdaloop/anipose/blob/master/anipose/label_videos_3d.py 
"""

from mayavi import mlab
mlab.options.offscreen = True

import numpy as np
from glob import glob
import pandas as pd
import os.path
from pathlib import Path
import cv2
import re
import toml

# import skvideo
# skvideo.setFFmpegPath('/beagle3/nicho/environments/mayavi/lib/python3.10/site-packages/ffmpeg/')

import skvideo.io
from tqdm import tqdm, trange

from collections import defaultdict
from matplotlib.pyplot import get_cmap
from subprocess import check_output


def project_3d_to_2d(points, cam_dict):
    points = points.reshape(-1, 1, 3)
    rvec = np.array(cam_dict['rotation'], dtype='float64').ravel()
    tvec = np.array(cam_dict['translation'], dtype='float64').ravel()
    dist = np.array(cam_dict['distortions'], dtype='float64').ravel()
    matrix = np.array(cam_dict['matrix'], dtype='float64')
    out, _ = cv2.fisheye.projectPoints(points, rvec, tvec, matrix, dist)
    return out

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_nframes(vidname):
    try:
        metadata = skvideo.io.ffprobe(vidname)
        length = int(metadata['video']['@nb_frames'])
        return length
    except KeyError:
        return 0

def get_video_params_cap(cap):
    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['nframes'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    params['fps'] = cap.get(cv2.CAP_PROP_FPS)
    return params

def get_video_params(fname):
    cap = cv2.VideoCapture(fname)
    params = get_video_params_cap(cap)
    cap.release()
    return params

def wc(filename):
    out = check_output(["wc", "-l", filename])
    num = out.decode('utf8').split(' ')[0]
    return int(num)

def get_data_length(fname):
    try:
        numlines = wc(fname) - 1
    except:
        if fname.suffix == 'csv':
            numlines = len(pd.read_csv(fname))
        elif fname.suffix == 'h5':
            numlines = len(pd.read_hdf(fname, key='position'))
    return numlines

def connect(points, bps, bp_dict, color):
    ixs = [bp_dict[bp] for bp in bps]
    return mlab.plot3d(points[ixs, 0], points[ixs, 1], points[ixs, 2],
                       np.ones(len(ixs)), reset_zoom=False,
                       color=color, tube_radius=None, line_width=10)

def connect_all(points, scheme, bp_dict, cmap):
    lines = []
    for i, bps in enumerate(scheme):
        line = connect(points, bps, bp_dict, color=cmap(i)[:3])
        lines.append(line)
    return lines

def update_line(line, points, bps, bp_dict):
    ixs = [bp_dict[bp] for bp in bps]
    # ixs = [bodyparts.index(bp) for bp in bps]
    new = np.vstack([points[ixs, 0], points[ixs, 1], points[ixs, 2]]).T
    line.mlab_source.points = new

def update_all_lines(lines, points, scheme, bp_dict):
    for line, bps in zip(lines, scheme):
        update_line(line, points, bps, bp_dict)


def connect_2d(img, points, bps, bodyparts, col=(0,255,0,255)):
    try:
        ixs = [bodyparts.index(bp) for bp in bps]
    except ValueError:
        return

    for a, b in zip(ixs, ixs[1:]):
        if np.any(np.isnan(points[[a,b]])):
            continue
        pa = tuple(np.int32(points[a]))
        pb = tuple(np.int32(points[b]))
        try:
            cv2.line(img, pa, pb, col, 4)
        except:
            tmp = []

def connect_all_2d(img, points, scheme, bodyparts, cmap):
    for cnum, bps in enumerate(scheme):
        col = cmap(cnum % 10, bytes=True)
        col = [int(c) for c in col]
        col[:3] = col[:3][::-1]
        connect_2d(img, points, bps, bodyparts, col)

def label_frame_2d(img, points, scheme, bodyparts, cmap):
    '''Adapted from https://github.com/lambdaloop/anipose/blob/master/anipose/label_videos.py#L36
    '''
    n_joints, _ = points.shape

    connect_all_2d(img, points, scheme, bodyparts, cmap)

    for lnum, (x, y) in enumerate(points):
        if np.isnan(x) or np.isnan(y):
            continue
        x = np.clip(x, 1, img.shape[1]-1)
        y = np.clip(y, 1, img.shape[0]-1)
        x = int(round(x))
        y = int(round(y))
        # col = cmap_c(lnum % 10, bytes=True)
        # col = [int(c) for c in col]
        col = (255, 255, 255)
        cv2.circle(img,(x,y), 7, col[:3], -1)

    return img

def visualize_labels(config, labels_fname, cam_fname, outname, view_side, cam_dict, params,):

    try:
        scheme = config['labeling']['scheme']
    except KeyError:
        scheme = []

    if labels_fname.suffix == '.csv':
        data = pd.read_csv(labels_fname)
    elif labels_fname.suffix == '.h5':
        data = pd.read_hdf(labels_fname, key='position')

    cols = [x for x in data.columns if '_error' in x]

    if len(scheme) == 0:
        bodyparts = [c.replace('_error', '') for c in cols]
    else:
        bodyparts = sorted(set([x for dx in scheme for x in dx]))

    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))

    all_points = np.array([np.array(data.loc[:, (bp+'_x', bp+'_y', bp+'_z')])
                           for bp in bodyparts], dtype='float64')

    all_errors = np.array([np.array(data.loc[:, bp+'_error'])
                           for bp in bodyparts], dtype='float64')

    all_scores = np.array([np.array(data.loc[:, bp+'_score'])
                           for bp in bodyparts], dtype='float64')

    all_ncams = np.array([np.array(data.loc[:, bp+'_ncams'])
                          for bp in bodyparts], dtype='float64')


    if config['triangulation']['optim']:
        all_errors[np.isnan(all_errors)] = 0
    else:
        all_errors[np.isnan(all_errors)] = 10000
    good = (all_errors < 100)
    all_points[~good] = np.nan

    not_enough_points = np.mean(all_ncams >= 2, axis=1) < 0.2
    all_points[not_enough_points] = np.nan

    all_points = all_points*params['factor_to_mm']
    
    all_points_flat = all_points.reshape(-1, 3)
    check = ~np.isnan(all_points_flat[:, 0])

    if np.sum(check) < 10:
        print('too few points to plot, skipping...')
        return

    low, high = np.percentile(all_points_flat[check], [5, 95], axis=0)

    nparts = len(bodyparts)
    nframes = all_points.shape[1]
    framedict = dict(zip(data['fnum'], data.index))

    writer = skvideo.io.FFmpegWriter(outname, inputdict={
        # '-hwaccel': 'auto',
        '-framerate': str(params['fps']),
    }, outputdict={
        '-vcodec': 'h264', '-qp': '28', '-pix_fmt': 'yuv420p'
    })

    if cam_fname is not None:
        reader = cv2.VideoCapture(cam_fname)
        if cam_dict is not None:
            M = np.identity(3)
            center = np.zeros(3)
            for i in range(3):
                center[i] = np.mean(data['center_{}'.format(i)])
                for j in range(3):
                    M[i, j] = np.mean(data['M_{}{}'.format(i, j)])
            all_points_flat_t = (all_points_flat + center).dot(np.linalg.inv(M.T))
            points_2d_proj = project_3d_to_2d(all_points_flat_t, cam_dict)
            points_2d_proj = points_2d_proj.reshape(nparts, nframes, 2)
  
    cmap = get_cmap('tab10')

    points = np.copy(all_points[:, 2000])
    points[0] = low
    points[1] = high

    s = np.arange(points.shape[0])
    good = ~np.isnan(points[:, 0])
    
    fig = mlab.figure(bgcolor=(1,1,1), size=(params['height'],params['height']))
    fig.scene.anti_aliasing_frames = 2

    low, high = np.percentile(points[good, 0], [10,90])
    scale_factor = (high - low) / 24

    mlab.clf()
    pts = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], s,
                        color=(0.8, 0.8, 0.8),
                        scale_mode='none', scale_factor=scale_factor)
    lines = connect_all(points, scheme, bp_dict, cmap)
    mlab.orientation_axes()

    if view_side == 'left':
        # view = [168, 68, 55, np.array([6.105, 1.664, 2.984])]
        view =[170.68694152055565,
               69.0085482680743,
               517.2005958747862,
               np.array([94.28086042, 47.94095707, 45.32973838])]

        # view = [168, 68]
        roll = 95
    elif view_side == 'right':
        view = [15, 68, 47, np.array([8.716, 0.839, 2.974])]
        roll = -95   
    else:
        view = list(mlab.view())
        roll = mlab.roll()
    
    mlab.view(*view)
    mlab.roll(roll)   
    # f = mlab.gcf()
    # f.scene._lift()
    # mlab.view(*view, focalpoint='auto', distance='auto')
    # mlab.roll(roll)
    
    view = list(mlab.view())

    for framenum in trange(data.shape[0], ncols=70):
        fig.scene.disable_render = True

        if framenum in framedict:
            points = all_points[:, framenum]
            if cam_dict is not None:
                proj_points = points_2d_proj[:, framenum]
        else:
            points = np.ones((nparts, 3))*np.nan
            if cam_dict is not None:
                proj_points = np.ones((nparts, 2))*np.nan

        s = np.arange(points.shape[0])
        good = ~np.isnan(points[:, 0])

        new = np.vstack([points[:, 0], points[:, 1], points[:, 2]]).T
        pts.mlab_source.points = new
        update_all_lines(lines, points, scheme, bp_dict)

        fig.scene.disable_render = False

        f = mlab.gcf()
        f.scene._lift()
        img = mlab.screenshot()
        
        if cam_fname is not None:
            _, frame  = reader.read()
            # frame = frame[...,::-1]
            if cam_dict is not None:
                frame = label_frame_2d(frame, proj_points, scheme, bodyparts, cmap)
            
            canvas = np.zeros((max(img.shape[0], frame.shape[0]),
                               img.shape[1] + frame.shape[1],
                               3), 
                              dtype = frame.dtype)
            canvas[:frame.shape[0], :frame.shape[1]] = frame[..., ::-1]
            canvas[:img.shape[0]  , frame.shape[1]:] = img
            img = canvas
        
        mlab.view(*view, reset_roll=False)

        writer.writeFrame(img)

        # if framenum > 100:
        #     break

    mlab.close(all=True)
    writer.close()

def process_session(config, session_path, view_side='left', post_processed=False, include_cam_vid=None, overwrite=False):
    pipeline_videos_raw = config['pipeline']['videos_raw']

    if post_processed:
        pipeline_videos_labeled_3d = 'videos-3d-post-processed'
        pipeline_3d = 'pose-3d-post-processed'
        data_ext = 'h5'
        factor_to_mm = 1e1
        if include_cam_vid is not None:
            calib_fname = session_path / 'calibration' / 'calibration.toml'
            calib_config = toml.load(calib_fname)
            cam_dict = [c_dict for key, c_dict in calib_config.items() if 'cam' in key and c_dict['name'] == str(include_cam_vid)][0]
        else:
            cam_dict = None
    else:
        pipeline_videos_labeled_3d = 'videos-3d'
        pipeline_3d = 'pose-3d'
        data_ext = 'csv'
        factor_to_mm=1
        cam_dict = None

    video_ext = config['video_extension']

    vid_fnames = sorted(list((session_path / pipeline_videos_raw).glob(f"*.{video_ext}")))
    orig_fnames = defaultdict(list)
    for vid in vid_fnames:
        cam_regex = config['triangulation']['cam_regex']
        vidname = re.sub(cam_regex, '', vid.stem).strip()
        orig_fnames[vidname].append(vid)

    labels_fnames = (session_path / pipeline_3d).glob(f'*.{data_ext}')
    labels_fnames = sorted([str(f) for f in labels_fnames], key=natural_keys)
    labels_fnames = [Path(f) for f in labels_fnames]

    if include_cam_vid is not None:
        if not post_processed:
            cam_vid_fnames = (session_path / 'videos-2d-proj').glob(f'*cam{include_cam_vid}*.mp4')
        else:
            cam_vid_fnames = (session_path / 'avi_videos').glob(f'*cam{include_cam_vid}*.avi')
        cam_vid_fnames = sorted([str(f) for f in cam_vid_fnames], key=natural_keys)
        cam_vid_fnames = [Path(f) for f in cam_vid_fnames]          
    else:
        cam_vid_fnames = [None for i in range(len(labels_fnames))]

    outdir = session_path / pipeline_videos_labeled_3d

    if len(labels_fnames) > 0:
        os.makedirs(outdir, exist_ok=True)

    for fname, cam_fname in zip(labels_fnames, cam_vid_fnames):

        out_fname = outdir / f'{fname.stem}.mp4'

        if not overwrite and out_fname.is_file() and \
           abs(get_nframes(out_fname) - get_data_length(fname)) < 100:
            continue
        print(out_fname)

        some_vid = orig_fnames[fname.stem][0]
        params = get_video_params(some_vid)
        params['factor_to_mm'] = factor_to_mm

        visualize_labels(config, fname, cam_fname, out_fname, view_side, cam_dict, params,)

if __name__ == '__main__':
    # date = '2021_02_10'
    # session_path = Path(f'/project/nicho/data/marmosets/kinematics_videos/moth/TYJL/{date}')
    date = '2023_08_04'
    session_path = Path(f'/project/nicho/data/marmosets/kinematics_videos/moth/JLTY/{date}')
    config = toml.load(session_path /'config.toml')
    process_session(config, session_path, post_processed=False, view_side='left', include_cam_vid=2, overwrite=True)