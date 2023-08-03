#!/bin/bash
#SBATCH --job-name=JL_free_video_proc
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/JL_free_video_proc_%a.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/JL_free_video_proc_%a.err
#SBATCH --time=36:00:00
#SBATCH --partition=caslake
#SBATCH --array=0-3
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --account=pi-nicho

#------------params that may change for each dataset/experiment----------------------#
dates=(2023_06_26)
exp=test_free
sess=(1)
calib_name=calib
scratch_space_username=daltonm

#----------------------params that rarely change-------------------------------------#
marms=HMMG
fps=60
ncams=4
vid_transpose=(2 2 1 1)  # number of arguments MUST match ncams 

#-----------paths to input and output data, and to processing scripts----------------#
jpg_dir=/scratch/midway3/$scratch_space_username/kinematics_jpgs
vid_dir=/project/nicho/data/marmosets/kinematics_videos
video_processing=/project/nicho/projects/marmosets/code_database/data_processing/kinematics/video_processing

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /project/nicho/environments/dlc

#---------------------Execute the programs-------------------------------------------#
python $video_processing/check_for_episode_splits_and_adjust_image_filenames_preAVI.py -j $jpg_dir -v $vid_dir -m $marms -d ${dates[@]} -e $exp -s ${sess[@]} -f $fps -n $ncams
python $video_processing/jpg2avi.py -j $jpg_dir -v $vid_dir -m $marms -d ${dates[@]} -e $exp -s ${sess[@]} -f $fps -n $ncams -t ${vid_transpose[@]} -c $calib_name
python $video_processing/apply_clahe_filter_to_videos.py -i $vid_dir/$exp/$marms/ -d ${dates[@]}
python $video_processing/recompress_filtered_avi.py      -i $vid_dir/$exp/$marms/ -d ${dates[@]}
