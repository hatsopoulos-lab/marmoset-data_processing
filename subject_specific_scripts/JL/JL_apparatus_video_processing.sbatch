#!/bin/bash
#SBATCH --job-name=JL_app_video_proc
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/Dalton08_redo2_JL_app_video_proc_%a.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/Dalton08_redo2_JL_app_video_proc_%a.err
#SBATCH --time=3:00:00
#SBATCH --partition=caslake
#SBATCH --array=0-19
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --account=pi-nicho

#------------params that may change for each dataset/experiment---------------#
dates=(2023_08_08)
exp1=moth
exp2=moth_free  # free
sess=(1)
calib_name=calib
touchscreen=False
scratch_space_username=daltonm #walkerjd


#----------------------params that rarely change------------------------------#
marms=JLTY
marms_ephys=JL
swap_ab=no
prb_path=/project/nicho/data/marmosets/prbfiles/JL_01.prb
metadata_path=/project/nicho/data/marmosets/metadata_yml_files/JL_complete_metadata.yml
fps=200 #150 #200
ncams=5 #2 #5
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
python $video_processing/check_for_episode_splits_and_adjust_image_filenames_preAVI.py \
-j $jpg_dir -v $vid_dir -m $marms -d ${dates[@]} -e $exp1 -s ${sess[@]} -f $fps -n $ncams

python $video_processing/jpg2avi.py -j $jpg_dir -v $vid_dir -m $marms -d ${dates[@]} -e $exp1 -s ${sess[@]} -f $fps -n $ncams -t ${vid_transpose[@]} -c $calib_name
python $video_processing/apply_clahe_filter_to_videos.py -i $vid_dir/$exp1/$marms/ -d ${dates[@]}
python $video_processing/recompress_filtered_avi.py      -i $vid_dir/$exp1/$marms/ -d ${dates[@]} 

source activate /project/nicho/environments/spyder
python $video_processing/process_analog_signals_for_episode_times.py \
-v $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d ${dates[@]} -e $exp1 -e2 $exp2 \
-t $touchscreen -tp $touchscreen_path -np $neural_processing -meta $metadata_path -prb $prb_path -ab $swap_ab

python $neural_processing/neural_dropout_first_pass.py -k $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d ${dates[@]} -e $exp1 -e2 $exp2