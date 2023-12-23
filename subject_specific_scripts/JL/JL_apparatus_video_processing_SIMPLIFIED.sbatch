#!/bin/bash
#SBATCH --job-name=JL15_app_video_proc
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/Sami15_JL_app_video_proc_%a.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/Sami15_JL_app_video_proc_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=caslake
#SBATCH --array=0-79 #0-19
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --account=pi-nicho
#SBATCH --mail-type=ARRAY_TASKS,FAIL
#SBATCH --mail-user=snjohnso@uchicago.edu

#------------params that may change for each dataset/experiment---------------#
date=2023_09_15
exp1=cricket
exp2=cricket_free  # free
sess=(1)
calib_name=calib
applyCLAHE=True
touchscreen=False
scratch_space_username=snjohnso


#----------------------params that rarely change------------------------------#
marms=JLTY
marms_ephys=JL
swap_ab=no
prb_path=/project/nicho/data/marmosets/prbfiles/JL_01.prb
metadata_path=/project/nicho/data/marmosets/metadata_yml_files/JL_complete_metadata.yml
fps=200 #150 #200
ncams=5 #2 #5
cams=(1 2 3 4 5)
vid_transpose=(-1 -1 -1 -1 -1) #(-1 -1) #(-1 -1 -1 -1 -1)  # number of arguments MUST match ncams 

#-----------paths to input and output data, and to processing scripts----------------#
jpg_dir=/scratch/midway3/$scratch_space_username/kinematics_jpgs
vid_dir=/project/nicho/data/marmosets/kinematics_videos
ephys_dir=/project/nicho/data/marmosets/electrophys_data_for_processing
video_processing=/project/nicho/projects/marmosets/code_database/data_processing/kinematics/video_processing
neural_processing=/project/nicho/projects/marmosets/code_database/data_processing/neural

touchscreen_path=BLANK

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /project/nicho/environments/dlc

#---------------------Execute the programs-------------------------------------------#
python $video_processing/check_for_episode_splits_and_adjust_image_filenames.py \
-j $jpg_dir -v $vid_dir -m $marms -d $date -e $exp1 -s ${sess[@]} -f $fps -n $ncams

python $video_processing/jpg2avi_bright.py -j $jpg_dir -v $vid_dir -m $marms -d $date \
-e $exp1 -s ${sess[@]} -f $fps -cm ${cams[@]} -t ${vid_transpose[@]} -c $calib_name -cl $applyCLAHE