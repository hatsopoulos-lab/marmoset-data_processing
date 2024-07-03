#!/bin/bash
#SBATCH --job-name=TY_0205_nwb_conv
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/TY_0205_nwb_conv.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/TY_0205_nwb_conv.err
#SBATCH --time=00:01:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#------------params that may change for each dataset/experiment----------------------#
dates=(2021_02_05 2021_02_06 2021_02_07 2021_02_08 2021_02_09 2021_02_10 2021_02_11 2021_02_12)
exp1=moth
exp2=free
sess=(1)
video_neural_session_alignment=matched #(matched, all_in_one_neural_recording)
touchscreen=False

#----------------------params that rarely change-------------------------------------#
marms=TYJL
marms_ephys=TY
swap_ab=yes
prb_path=/project/nicho/data/marmosets/prbfiles/TY_02.prb
metadata_path=/project/nicho/data/marmosets/metadata_yml_files/TY_complete_metadata.yml
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
source activate /beagle3/nicho/environments/nwb_and_neuroconv

#---------------------------------------------Execute the programs------------------------#

for date in ${dates[@]}
do 
    echo $date
done