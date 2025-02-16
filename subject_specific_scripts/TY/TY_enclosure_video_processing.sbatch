#!/bin/bash
#SBATCH --job-name=TY0125_free_video_proc_b3
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/Sami0125_TY_free_b3_video_proc_%a.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/Sami0125_TY_free_b3_video_proc_%a.err
#SBATCH --time=36:00:00
#SBATCH --partition=beagle3
#SBATCH --gres=gpu:1
#SBATCH --array=0-3 #0-11, minimum 4
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --account=pi-nicho
#SBATCH --mail-type=ARRAY_TASKS,FAIL
#SBATCH --mail-user=snjohnso@uchicago.edu

#------------params that may change for each dataset/experiment----------------------#
date=2025_01_25
exp1=baseline
exp2=baselineFree
sess=(1)
video_neural_session_alignment=matched #(matched, all_in_one_neural_recording)
calib_name=calib
applyCLAHE=True
touchscreen=False
scratch_space_username=snjohnso

#----------------------params that rarely change-------------------------------------#
marms=TYTR
fps=30
ncams=4
cams=(1 2 3 4)
vid_transpose=(2 2 1 1)  # number of arguments MUST match ncams 
analog_signals_fps=(150 $fps)

marms_ephys=TY
swap_ab=no 
prb_path=/project/nicho/data/marmosets/prbfiles/TY_02.prb
metadata_path=/project/nicho/data/marmosets/metadata_yml_files/TY_complete_metadata.yml

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
python $video_processing/check_for_episode_splits_and_adjust_image_filenames.py -j $jpg_dir \
-v $vid_dir -m $marms -d $date -e $exp2 -s ${sess[@]} -f $fps -n $ncams

python $video_processing/jpg2avi.py -j $jpg_dir -v $vid_dir -m $marms -d $date \
-e $exp2 -s ${sess[@]} -f $fps -cm ${cams[@]} -t ${vid_transpose[@]} -c $calib_name -cl $applyCLAHE

source activate /beagle3/nicho/environments/nwb_and_neuroconv

python $video_processing/process_analog_signals_for_episode_times.py \
-v $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d $date -e $exp1 -e2 $exp2 \
-t $touchscreen -tp $touchscreen_path -np $neural_processing -meta $metadata_path \
-prb $prb_path -ab $swap_ab -al $video_neural_session_alignment -f ${analog_signals_fps[@]}

python $neural_processing/neural_dropout_first_pass.py -k $vid_dir -ep $ephys_dir -m $marms \
-me $marms_ephys -d $date -e $exp1 -e2 $exp2
