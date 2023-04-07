#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 18:41:09 2022

@author: daltonm
"""
                                                                  
import glob
import os
import argparse
import shutil   
import re
import pandas as pd


event_pattern = re.compile('e\d{3}')
sess_pattern  = re.compile('_s\d{1}')
long_event_pattern = re.compile('event\d{3}')
long_sess_pattern  = re.compile('session\d{1}')
cam_pattern   = re.compile('cams\d{1}_and_\d{1}')

def dlc_to_xma(cam1data,cam2data,trialname,savepath):
    # Adapted from XROMM_DLC_tools by J.D. Laurence-Chasen
    
    # h5_save_path = savepath+"/"+trialname+"-Predicted2DPoints.h5"
    csv_save_path = os.path.join(savepath, trialname+"-Predicted2DPoints.csv")
    
    if isinstance(cam1data, str): #is string
        if ".csv" in cam1data:

            cam1data=pd.read_csv(cam1data, sep=',',header=None)
            cam2data=pd.read_csv(cam2data, sep=',',header=None)
            pointnames = list(cam1data.loc[1,1:].unique())
            
            # reformat CSV / get rid of headers
            cam1data = cam1data.loc[3:,1:]
            cam1data.columns = range(cam1data.shape[1])
            cam1data.index = range(cam1data.shape[0])
            cam2data = cam2data.loc[3:,1:]
            cam2data.columns = range(cam2data.shape[1])
            cam2data.index = range(cam2data.shape[0])
            
        elif ".h5" in cam1data:# is .h5 file
            cam1data = pd.read_hdf(cam1data)
            cam2data = pd.read_hdf(cam2data)
            pointnames = list(cam1data.columns.get_level_values('bodyparts').unique())

        else:
            raise ValueError('2D point input is not in correct format')
    else:
        
        pointnames = list(cam1data.columns.get_level_values('bodyparts').unique())
    
    # make new column names
    nvar = len(pointnames)
    pointnames = [item for item in pointnames for repetitions in range(4)]
    post = ["_cam1_X", "_cam1_Y", "_cam2_X", "_cam2_Y"]*nvar
    cols = [m+str(n) for m,n in zip(pointnames,post)]


    # remove likelihood columns
    cam1data = cam1data.drop(cam1data.columns[2::3],axis=1)
    cam2data = cam2data.drop(cam2data.columns[2::3],axis=1)

    # replace col names with new indices
    c1cols = list(range(0,cam1data.shape[1]*2,4)) + list(range(1,cam1data.shape[1]*2,4))
    c2cols = list(range(2,cam1data.shape[1]*2,4)) + list(range(3,cam1data.shape[1]*2,4))
    c1cols.sort()
    c2cols.sort()
    cam1data.columns = c1cols
    cam2data.columns = c2cols
    df = pd.concat([cam1data,cam2data],axis=1).sort_index(axis=1)
    df.columns = cols
    # df.to_hdf(h5_save_path, key="df_with_missing", mode="w")
    df.to_csv(csv_save_path,na_rep='NaN',index=False)


def copy_files_to_corrections_folder(dirs, scorer, args):
     
    episodes = args['episodes'] 
    cams     = args['cam_pair']
    session  = args['session']
    
    pose_files = sorted(glob.glob(os.path.join(dirs['pose'], '*')))
    try:
        pose_files = [f for f in pose_files 
                      if  int(re.findall(sess_pattern , f)[0].split('s'  )[-1]) == session
                      and int(re.findall(event_pattern, f)[0].split('e'  )[-1]) in episodes
                      and int(re.findall(cam_pattern  , f)[0].split('cam')[-1]) in cams]
    except:
        pose_files = [f for f in pose_files 
                      if  int(re.findall(long_sess_pattern , f)[0].split('session')[-1]) == session
                      and int(re.findall(long_event_pattern, f)[0].split('event'  )[-1]) in episodes
                      and int(re.findall(cam_pattern       , f)[0].split('cam'    )[-1]) in cams]
        
    correction_files = [0]*len(pose_files)
    for idx, f in enumerate(pose_files):
        base = os.path.basename(f)
        base, ext = os.path.splitext(base)
        
        corr_f = os.path.join(dirs['corrections_orig'], base + scorer + ext)
        shutil.copy(f, corr_f)
        correction_files[idx] = corr_f
        
        shutil.copy(os.path.join(dirs['video'], base + '.avi'),
                    os.path.join(dirs['corrections_vids']))
    
    cam1_files = [f for f in correction_files if int(re.findall(cam_pattern, f)[0].split('cam')[-1]) == cams[0]]
    cam2_files = [f for f in correction_files if int(re.findall(cam_pattern, f)[0].split('cam')[-1]) == cams[1]]    
    
    for cam1data, cam2data in zip(cam1_files, cam2_files):
        trialname = os.path.basename(cam1data)
        trialname, ext = os.path.splitext(trialname)
        
        cam_text = re.search(cam_pattern, trialname)[0]
        trialname = trialname.replace(cam_text, 'cams_%d_and_%d' % (cams[0], cams[1]))
        
        dlc_to_xma(cam1data, cam2data, trialname, dirs['dlc_to_xma'])
        
        
        
    
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--pose_dir", required=True,
    #     help="path to directory holding .h5 and .pickle files. E.g. '/project/nicho/data/marmosets/kinematics_videos/test/TYJL/2022_06_17/pose-2d'")
    # ap.add_argument("-v", "--vid_dir", required=True,
    #     help="path to directory holding origina .avi files. E.g. '/project/nicho/data/marmosets/kinematics_videos/test/TYJL/2022_06_17/avi_videos'")

    # args = vars(ap.parse_args())
    
    args = {'pose_dir': '/project/nicho/data/marmosets/kinematics_videos/moths/TYJL/2021_02_11/pose-2d-proj',
            'session' : 1,
            'episodes': [85, 113, 146],
            'cam_pair': [1, 2]}
      
    pose_dir = args['pose_dir']
    if pose_dir[-1] == '/':
        pose_dir = pose_dir[:-1]
    
    dirs = {'pose' : pose_dir}
    
    basedir = os.path.split(pose_dir)[0]
    dirs['video'] = os.path.join(basedir, 'avi_videos')    
    dirs['corrections_orig'] = os.path.join(basedir, 'xmalab_corrections', 'original_trajectories')
    dirs['corrections_corr'] = os.path.join(basedir, 'xmalab_corrections', 'corrected_trajectories')
    dirs['corrections_vids'] = os.path.join(basedir, 'xmalab_corrections', 'videos')
    dirs['dlc_to_xma'] = os.path.join(basedir, 'xmalab_corrections', 'dlc_to_xma')
    dirs['xma_to_dlc'] = os.path.join(basedir, 'xmalab_corrections', 'xma_to_dlc')
  
    os.makedirs(dirs['corrections_orig'], exist_ok=True)
    os.makedirs(dirs['corrections_corr'], exist_ok=True)
    os.makedirs(dirs['corrections_vids'], exist_ok=True)
    os.makedirs(dirs['dlc_to_xma'], exist_ok=True)
    os.makedirs(dirs['xma_to_dlc'], exist_ok=True)

    with open(os.path.join(basedir, 'scorer_info.txt')) as f:
        scorer = f.readlines()[0]
        
    scorer = scorer.split('filtered')[-1]
    scorer = scorer.split('_meta')[0]
    
    copy_files_to_corrections_folder(dirs, scorer, args)