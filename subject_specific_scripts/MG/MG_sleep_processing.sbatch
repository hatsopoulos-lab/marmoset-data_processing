#!/bin/bash
#SBATCH --job-name=sleep_video_proc
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/sleep_video_proc.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/sleep_video_proc.err
#SBATCH --time=06:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#------------params that may change for each dataset/experiment----------------------#
dates=(2023_05_01)
exp=sleep
sess=(1)
calib_name=None
touchscreen=False
scratch_space_username=ari798

other_exp=None

#----------------------params that rarely change-------------------------------------#
marms=HMMG
marms_ephys=MG
swap_ab=no
prb_path=/project/nicho/data/marmosets/prbfiles/MG_01.prb
metadata_path=/project/nicho/data/marmosets/metadata_yml_files/MG_complete_metadata.yml
fps=30
ncams=1
vid_transpose=(1) # number of arguments MUST match ncams 

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
python $video_processing/check_for_episode_splits_and_adjust_image_filenames_preAVI.py -j $jpg_dir -v $vid_dir -m $marms -d ${dates[@]} -e $exp -s ${sess[@]} -f $fps -n $ncams
python $video_processing/jpg2avi_sleep.py -j $jpg_dir -v $vid_dir -m $marms -d ${dates[@]} -e $exp -s ${sess[@]} -f $fps -n $ncams -t ${vid_transpose[@]} -c $calib_name
python $video_processing/apply_clahe_filter_to_videos_sleep.py -i $vid_dir/$exp/$marms/ -d ${dates[@]}
python $video_processing/recompress_filtered_avi_sleep.py      -i $vid_dir/$exp/$marms/ -d ${dates[@]} 

source activate /project/nicho/environments/spyder
python $video_processing/process_analog_signals_for_episode_times.py \
-v $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d ${dates[@]} -e $exp1 -e2 $exp2 \
-t $touchscreen -tp $touchscreen_path -np $neural_processing -meta $metadata_path -prb $prb_path -ab $swap_ab

python $neural_processing/neural_dropout_first_pass.py -k $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d ${dates[@]} -e $exp -e2 $other_exp
