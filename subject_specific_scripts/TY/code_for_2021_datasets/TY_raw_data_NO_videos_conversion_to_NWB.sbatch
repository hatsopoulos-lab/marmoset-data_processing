#!/bin/bash
#SBATCH --job-name=TY_sigCheck_nwb_conv
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/TY_sigCheck_nwb_conv.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/TY_sigCheck_nwb_conv.err
#SBATCH --time=2:00:00
#SBATCH --partition=beagle3
#SBATCH --array=0
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#------------params that may change for each dataset/experiment----------------------#
dates=(2021_03_09 2021_03_13 2021_03_14) #(2021_03_12 2021_03_13 2021_03_14 2021_03_15 2021_03_16 2021_03_17 2021_03_18 2021_03_19 2021_03_20 2021_03_21) # (2021_03_08 2021_03_09 2021_03_10 2021_03_11) # 2021_01_27 2021_01_28 2021_01_29 2021_01_30 2021_01_31 2021_02_01 2021_02_03 2021_02_04 2021_02_05 2021_02_06 2021_02_07 2021_02_08 2021_02_09 2021_02_10 2021_02_11 2021_02_12 2021_02_13 2021_02_14 2021_02_15 2021_02_16 2021_02_17 2021_02_18 2021_02_19 2021_02_22 2021_02_23 2021_02_24 2021_02_25 2021_03_01 2021_03_03 2021_03_04 2021_03_06 2021_03_07 2021_03_08 2021_03_09 2021_03_10 2021_03_11 2021_03_12 2021_03_13 2021_03_14 2021_03_15 2021_03_16 2021_03_17 2021_03_18 2021_03_19 2021_03_20 2021_03_21 2021_03_26 2021_03_27 2021_03_28 2021_03_29 2021_03_30 2021_03_31 2021_04_01 2021_04_02 2021_04_03 2021_04_04)
exp1=temp
exp2=signalCheck

#----------------------params that rarely change-------------------------------------#
marms=TYJL
marms_ephys=TY
swap_ab=yes
prb_path=/project/nicho/data/marmosets/prbfiles/TY_02.prb
metadata_path=/project/nicho/data/marmosets/metadata_yml_files/TY_complete_metadata.yml

#-----------paths to input and output data, and to processing scripts----------------#
vid_dir=/project/nicho/data/marmosets/kinematics_videos
ephys_dir=/project/nicho/data/marmosets/electrophys_data_for_processing/NO_NWB
video_processing=/project/nicho/projects/marmosets/code_database/data_processing/kinematics/video_processing
neural_processing=/project/nicho/projects/marmosets/code_database/data_processing/neural

touchscreen_path=BLANK

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 

#---------------------------------------------Execute the programs------------------------#

source activate /beagle3/nicho/environments/nwb_and_neuroconv

for date in ${dates[@]}
do
    python $video_processing/organize_files_for_nwb_creation_in_noVideo_datasets.py \
    -v $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d $date -e $exp1 -e2 $exp2 \
    -np $neural_processing -meta $metadata_path \
    -prb $prb_path -ab $swap_ab

    python $neural_processing/neural_dropout_first_pass.py -k $vid_dir -ep $ephys_dir -m $marms \
    -me $marms_ephys -d $date -e $exp1 -e2 $exp2
done