# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 04:49:17 2021

@author: Dalton
"""
import argparse
import subprocess

class params:
    aniposePath = '/media/marms/DATA/anipose_dlc_validation_paper'
    2d_filters = ['viterbi', 'median', 'autoencoder']

def change_param(configPath, oldStr, newStr, subset):
    file = open(configPath, "rt")
    data = file.read()
    splitStr = 'video_sets'
    projText, vidText = data.split(splitStr)
    if subset == 'project':
        projText = projText.replace(oldStr, newStr)
        data = projText + splitStr + vidText
    elif subset == 'video':
        vidText = vidText.replace(oldStr, newStr)
        data = projText + splitStr + vidText
    elif subset == 'all':
        data = data.replace(oldStr, newStr)
    else:
        print('Please enter an allowed subset of paths to replace. Options are "project", "video", or "all" (default)')
    
    file.close()
    file = open(configPath, "wt")
    file.write(data)
    file.close()

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config_path", required=True,
     	help="path to dlc config.yaml file")
    ap.add_argument("-o", "--old_path",
     	help="current prefix to project and video paths. E.g. 'Z:'")
    ap.add_argument("-n", "--new_path",
     	help="new prefix. E.g. '/gpfs/data/nicho-lab'")
    ap.add_argument("-s", "--subset_to_replace", default = 'all', 
        help="whether to change project_path, video_set paths, or all. Input can be 'project', 'video', or 'all'. Default = 'all'")
    args = vars(ap.parse_args())
    
    replace_path(args['config_path'], args['old_path'], args['new_path'], args['subset_to_replace'])
    