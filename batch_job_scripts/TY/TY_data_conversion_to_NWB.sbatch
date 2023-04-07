#!/bin/bash
#SBATCH --job-name=TY_nwb_conv
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/TY_nwb_conv.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/TY_nwb_conv.err
#SBATCH --time=01:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#------------params that may change for each dataset/experiment----------------------#
dates=(2021_02_11)
exp1=moths
exp2=free
touchscreen=False

#----------------------params that rarely change-------------------------------------#
marms=TYJL
marms_ephys=TY
prb_path=/project/nicho/data/marmosets/prbfiles/TY_02.prb
metadata_path=/project/nicho/projects/marmosets/code_database/data_processing/nwb_tools/TY_complete_metadata.yml
ncams_1=2
ncams_2=2
vid_transpose_1=(-1 -1)  # number of arguments MUST match ncams 
vid_transpose_2=(2 2)

#-----------paths to input and output data, and to processing scripts----------------#
vid_dir=/project/nicho/data/marmosets/kinematics_videos
ephys_dir=/project/nicho/data/marmosets/electrophys_data_for_processing
video_processing=/project/nicho/projects/marmosets/code_database/data_processing/kinematics/video_processing
neural_processing=/project/nicho/projects/marmosets/code_database/data_processing/neural
calib_base_path_1=/project/nicho/data/marmosets/kinematics_videos/$exp1/$marms
calib_base_path_2=/project/nicho/data/marmosets/kinematics_videos/$exp2/$marms

touchscreen_path=BLANK

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /project/nicho/environments/dlc

#---------------------------------------------Execute the programs------------------------#
python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/check_for_and_create_calibration_videos.py \
-a $calib_base_path_1 -t ${vid_transpose_1[@]} -n $ncams_1
python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/check_for_and_create_calibration_videos.py \
-a $calib_base_path_2 -t ${vid_transpose_2[@]} -n $ncams_2

source activate /project/nicho/environments/spyder

python $video_processing/process_analog_signals_for_episode_times_TonyData.py \
-v $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d ${dates[@]} -e $exp1 -e2 $exp2 -t $touchscreen -tp $touchscreen_path -np $neural_processing -meta $metadata_path -prb $prb_path

python $neural_processing/neural_dropout_first_pass.py -k $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d ${dates[@]} -e $exp1 -e2 $exp2
