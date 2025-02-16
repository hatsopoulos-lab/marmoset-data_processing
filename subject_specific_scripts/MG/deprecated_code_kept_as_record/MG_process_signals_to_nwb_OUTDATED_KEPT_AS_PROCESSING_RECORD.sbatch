#!/bin/bash
#SBATCH --job-name=MG_nwb_conv
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/MG_nwb_conv.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/MG_nwb_conv.err
#SBATCH --time=00:02:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --account=pi-nicho
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=daltonm@uchicago.edu

#------------params that may change for each dataset/experiment---------------#
date=2023_04_16
exp1=moths
exp2=free
sess=(1)
calib_name=calib
touchscreen=False
scratch_space_username=daltonm


#----------------------params that rarely change------------------------------#
marms=HMMG
marms_ephys=MG
swap_ab=no
prb_path=/project/nicho/data/marmosets/prbfiles/MG_01.prb
metadata_path=/project/nicho/data/marmosets/metadata_yml_files/MG_complete_metadata.yml
fps=200
ncams=5
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
source activate neuroconv_environment

#---------------------Execute the programs-------------------------------------------#
python $video_processing/process_analog_signals_for_episode_times.py \
-v $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d $date -e $exp1 -e2 $exp2 \
-t $touchscreen -tp $touchscreen_path -np $neural_processing -meta $metadata_path -prb $prb_path -ab $swap_ab

#python $neural_processing/neural_dropout_first_pass.py -k $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d $date -e $exp1 -e2 $exp2
