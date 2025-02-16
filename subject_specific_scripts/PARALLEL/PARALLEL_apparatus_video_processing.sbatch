#!/bin/bash
#SBATCH --job-name=P1111_app_video_proc_b3
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/Sami1111_PARALLEL_app_b3_video_proc_%a.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/Sami1111_PARALLEL_app_b3_video_proc_%a.err
#SBATCH --time=6:00:00
#SBATCH --partition=beagle3
#SBATCH --gres=gpu:1
#SBATCH --array=0-4 #0-19 or 79, minimum 5
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --account=pi-nicho
#SBATCH --mail-type=ARRAY_TASKS,FAIL
#SBATCH --mail-user=snjohnso@uchicago.edu

#------------params that may change for each dataset/experiment---------------#
date=2024_11_11
exp1=test
exp2=testFree
sess=(1)
calib_name=calib
applyCLAHE=True
touchscreen=False
scratch_space_username=snjohnso


#----------------------params that rarely change------------------------------#
marms=TYJL
marms_ephys=TYJL
swap_ab=no
prb_path1=/project/nicho/data/marmosets/prbfiles/parallel_recordings/TY_02_parallel.prb
prb_path2=/project/nicho/data/marmosets/prbfiles/parallel_recordings/JL_01_parallel.prb
metadata_path1=/project/nicho/data/marmosets/metadata_yml_files/parallel_recordings/TY_metadata_parallel.yml
metadata_path2=/project/nicho/data/marmosets/metadata_yml_files/parallel_recordings/JL_metadata_parallel.yml
fps=150 #150 #200
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
module load python/anaconda-2023.09 
source activate /project/nicho/environments/dlc

#---------------------Execute the programs-------------------------------------------#
python $video_processing/check_for_episode_splits_and_adjust_image_filenames.py \
-j $jpg_dir -v $vid_dir -m $marms -d $date -e $exp1 -s ${sess[@]} -f $fps -n $ncams

python $video_processing/jpg2avi.py -j $jpg_dir -v $vid_dir -m $marms -d $date \
-e $exp1 -s ${sess[@]} -f $fps -cm ${cams[@]} -t ${vid_transpose[@]} -c $calib_name -cl $applyCLAHE