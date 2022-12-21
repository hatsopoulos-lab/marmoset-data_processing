# -*- coding: utf-8 -*-
"""
Created on June 07 2022

@author: Dalton
"""

# An automated processing script for converting jpg files into videos.
# An automated processing script for converting jpg files into videos.
                                                                     
# Example: sudo Documents/camera_control_code/jpg2avi_apparatus.sh 'TYJL' '2021_01_08' 'foraging' '1' '150'

#                                                                marmCode    date      expName session  framerate

import glob
import re
import os
import subprocess
import argparse
import time
import numpy as np

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--jpg_dir", required=True, type=str, nargs='+',
        help="path to temporary directory holding jpg files for task and marmoset pair. E.g. /scratch/midway3/daltonm/kinematics_jpgs/")
    ap.add_argument("-s", "--session", required=True, type=int, nargs='+',
        help="path to directory for task and marmoset pair. E.g. /project/nicho/data/marmosets/kinematics_videos/")

    args = vars(ap.parse_args())
    
    print((args['jpg_dir'], args['jpg_dir'][0], type(args['jpg_dir'][0])))
    print((args['session'], args['session'][0], type(args['session'][0])))

