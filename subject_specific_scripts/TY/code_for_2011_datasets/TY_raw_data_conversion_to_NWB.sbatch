#!/bin/bash
#SBATCH --job-name=crickets_nwb_conv
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/TY_cricket_nwb_conv.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/TY_cricket_nwb_conv.err
#SBATCH --time=8:00:00
#SBATCH --partition=beagle3
#SBATCH --array=0
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#------------params that may change for each dataset/experiment----------------------#
#dates=(2021_03_30) exp2=BeTL CHANGE TO avi_videos
#dates=(2021_03_07 2021_03_08 2021_03_09 2021_03_10 2021_03_11 2021_03_12 2021_03_13 2021_03_14 2021_03_15 2021_03_16 2021_03_17 2021_03_18 2021_03_19 2021_03_20 2021_03_21)
#dates=(2021_03_31 2021_04_01 2021_04_02 2021_04_03 2021_04_04) #(2021_01_09 2021_01_10 2021_01_11 2021_01_12 2021_01_13 2021_01_18 2021_01_19 2021_01_20 2021_01_22 2021_01_23 2021_01_24 2021_01_25 2021_03_01 2021_03_02 2021_03_03 2021_03_04 2021_03_26 2021_03_27 2021_03_28 2021_03_29 2021_03_30 2021_03_31 2021_04_01 2021_04_02 2021_04_03 2021_04_04)
dates=(2021_03_11 2021_03_12)
exp1=free
exp2=cricket
video_neural_session_alignment=all_in_one_neural_recording #(matched, all_in_one_neural_recording)
touchscreen=False

#----------------------params that rarely change-------------------------------------#
marms=TYJL
marms_ephys=TY
swap_ab=yes
prb_path=/project/nicho/data/marmosets/prbfiles/TY_02.prb
metadata_path=/project/nicho/data/marmosets/metadata_yml_files/TY_complete_metadata.yml
analog_signals_fps=(150 30)

#-----------paths to input and output data, and to processing scripts----------------#
vid_dir=/project/nicho/data/marmosets/kinematics_videos
ephys_dir=/project/nicho/data/marmosets/electrophys_data_for_processing
video_processing=/project/nicho/projects/marmosets/code_database/data_processing/kinematics/video_processing
neural_processing=/project/nicho/projects/marmosets/code_database/data_processing/neural

touchscreen_path=BLANK

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 

#---------------------------------------------Execute the programs------------------------#

source activate /project/nicho/environments/dlc

#vid_transpose=(-1 -1)
#python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/check_for_and_create_calibration_videos.py \
#-a /project/nicho/data/marmosets/kinematics_videos/$exp1/$marms -t ${vid_transpose[@]} -n 2

#vid_transpose=(2 2)
#python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/check_for_and_create_calibration_videos.py \
#-a /project/nicho/data/marmosets/kinematics_videos/$exp2/$marms -t ${vid_transpose[@]} -n 2

source activate /beagle3/nicho/environments/nwb_and_neuroconv

for date in ${dates[@]}
do
    python /project/nicho/projects/marmosets/code_database/data_processing/subject_specific_scripts/TY/code_for_2011_datasets/TY_process_analog_signals_for_episode_times.py \
    -v $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d $date -e $exp1 -e2 $exp2 \
    -t $touchscreen -tp $touchscreen_path -np $neural_processing -meta $metadata_path \
    -prb $prb_path -ab $swap_ab -al $video_neural_session_alignment -f ${analog_signals_fps[@]}

    python $neural_processing/neural_dropout_first_pass.py -k $vid_dir -ep $ephys_dir -m $marms \
    -me $marms_ephys -d $date -e $exp1 -e2 $exp2
done