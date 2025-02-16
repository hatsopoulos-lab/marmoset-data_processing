# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:54:29 2021

@author: Dalton
"""

# import dill
import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import dill
# import h5py
import matplotlib
import matplotlib.pyplot as plt
# from statsmodels.stats.weightstats import DescrStatsW
# from sklearn.decomposition import PCA
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter # median_filter
from scipy.signal import find_peaks, savgol_filter #peak_prominences, peak_widths
from scipy.spatial.distance import euclidean
# from scipy.stats import mode
# from scipy.interpolate import interp1d
from importlib import sys
import seaborn as sns

sys.path.insert(0, '/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/hatlab_nwb_tools/')
from hatlab_nwb_functions import load_dict_from_hdf5

marm = 'JL'
fps=200

save_plots=True

grasp_path = Path(f'/project/nicho/projects/dalton/plots_for_nicho/{marm}_kinematics/grasp')
limb_pos_path = Path(f'/project/nicho/projects/dalton/plots_for_nicho/{marm}_kinematics/full_limb_position')

anipose_base = '/project/nicho/data/marmosets/kinematics_videos/moth/JLTY/'

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(context='notebook', style="ticks", palette='Dark2', rc=custom_params)
dark2 = sns.color_palette("Dark2")

class dpath:
    base = Path(anipose_base)
    date = '2023_08_04'
    reach_data_path = Path('/project/nicho/data/marmosets/processed_datasets/reach_and_trajectory_information') / f'{date.replace("_","")}_reach_and_trajectory_info.h5'
        
    

def plot_xyz_of_wrist_and_hand(reach_data, event_nums= None, markers = ['l-wrist', 'l-d2-mcp', 'l-d5-mcp'], save_path = None):
    
    if event_nums is None:
        event_nums= sorted([event_data['event'] for event_data in reach_data])
        
    for event_data in reach_data:
        if event_data['event'] not in event_nums:
            continue
        
        fig, axs = plt.subplots(3, 1, figsize=(14,9), sharex=True)
        
        timestamps = np.linspace(0, event_data['position'].shape[-1] / fps, event_data['position'].shape[-1])
        for dim, dimLabel in enumerate(['x', 'y', 'z']):
            for start, stop in zip(event_data["starts"], event_data["stops"]):
                axs[dim].axvspan(timestamps[start], timestamps[stop], color='k', alpha=0.25)
            for mLabel in markers:
                mIdx = [idx for idx, val in enumerate(reach_data[0]["marker_names"]) if val.decode() == mLabel][0]
                axs[dim].plot(timestamps, event_data['position'][mIdx, dim, :], label=mLabel)
            axs[dim].set_ylabel(f'{dimLabel} (cm)')
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axs[2].set_xlabel('Time (sec)')
        fig.suptitle(f'Event {event_data["event"]}')

        if save_plots:
            fig.savefig(save_path / f"{marm}_event_{event_data['event']}.png")
                        
        plt.show()
        

def plot_grasp_and_splay(reach_data, event_nums= None, arm = 'l', save_path=None):
    
    if event_nums is None:
        event_nums= sorted([event_data['event'] for event_data in reach_data])


    d2_dip_idx = [idx for idx, val in enumerate(reach_data[0]["marker_names"]) if val.decode() == f'{arm}-d2-dip'][0]
    d5_dip_idx = [idx for idx, val in enumerate(reach_data[0]["marker_names"]) if val.decode() == f'{arm}-d5-dip'][0]
    d2_mcp_idx = [idx for idx, val in enumerate(reach_data[0]["marker_names"]) if val.decode() == f'{arm}-d2-mcp'][0]
    d5_mcp_idx = [idx for idx, val in enumerate(reach_data[0]["marker_names"]) if val.decode() == f'{arm}-d5-mcp'][0]
    wrist_idx  = [idx for idx, val in enumerate(reach_data[0]["marker_names"]) if val.decode() == f'{arm}-wrist' ][0]
    
    for event_data in reach_data:
        if event_data['event'] not in event_nums:
            continue
        
        pos = event_data['position']
        
        d2_extension = np.full((pos.shape[-1],), np.nan)
        d5_extension = np.full((pos.shape[-1],), np.nan)
        avg_extension = np.full((pos.shape[-1],), np.nan)
        splay = np.full((pos.shape[-1],), np.nan)
        for sample in range(pos.shape[-1]):
            try:
                d2_extension[sample] = euclidean(pos[d2_dip_idx, :, sample], pos[wrist_idx, :, sample])
            except:
                pass
            try:
                d5_extension[sample] = euclidean(pos[d5_dip_idx, :, sample], pos[wrist_idx, :, sample])
            except:
                pass
            try:
                avg_extension[sample] = (d5_extension[sample] + d2_extension[sample])/2
            except:
                pass
            try:
                splay[sample] = euclidean(pos[d2_dip_idx, :, sample], pos[d5_dip_idx, :, sample])
            except:
                pass
            
        
        timestamps = np.linspace(0, pos.shape[-1] / fps, pos.shape[-1])
        
        for reach_idx, (start, stop) in enumerate(zip(event_data["starts"], event_data["stops"])):
            fig, axs = plt.subplots(2, 1, figsize=(14,6), sharex=True)

            plot_idxs = np.where((timestamps > timestamps[start]-1) & (timestamps < timestamps[stop]+1))[0]            
            for ax in axs:
                ax.axvspan(timestamps[start], timestamps[stop], color='k', alpha=0.25)

            axs[0].plot(timestamps[plot_idxs], d2_extension[plot_idxs], label='Index Extension')      
            axs[0].plot(timestamps[plot_idxs], d5_extension[plot_idxs], label='Pinky Extension')   
            # axs[0].plot(timestamps[plot_idxs], avg_extension[plot_idxs], label='Extension')      

            axs[0].plot(timestamps[plot_idxs], splay[plot_idxs],  label='Splay')        
            axs[0].set_ylim(0, 4)
            for dim, dimLabel in enumerate(['x', 'y', 'z']):
                axs[1].plot(timestamps[plot_idxs], pos[wrist_idx, dim, plot_idxs], color=dark2[3+dim], label=f'wrist {dimLabel}')
                axs[1].set_ylabel(f'Wrist Position (cm)')
                
            axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
            axs[1].set_xlabel('Time (sec)')
                
            fig.suptitle(f'Event {event_data["event"]} - Reach {reach_idx+1}')
            
            if save_plots:
                fig.savefig(save_path / f'{marm}_event_{event_data["event"]}_reach_{reach_idx+1}.png')
            
        plt.show()

if __name__ == "__main__":
 
    reach_data = load_dict_from_hdf5(dpath.reach_data_path, top_level_list = True)

    event_nums = None

    os.makedirs(grasp_path, exist_ok=True)
    os.makedirs(limb_pos_path, exist_ok=True)

    plot_xyz_of_wrist_and_hand(reach_data, event_nums=event_nums, markers = ['l-wrist', 'l-d2-mcp', 'l-d5-mcp', 'l-d2-dip', 'l-d5-dip', 'l-elbow', 'l-shoulder'], save_path = limb_pos_path)
    
    plot_grasp_and_splay(reach_data, event_nums=event_nums, arm = 'l', save_path = grasp_path)
    
