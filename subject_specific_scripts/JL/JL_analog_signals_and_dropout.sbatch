#!/bin/bash
#SBATCH --job-name=0809_analog
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/Dalton0809_analog.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/Dalton0809_analog.err
#SBATCH --time=4:00:00
#SBATCH --partition=caslake
#SBATCH --array=0
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --account=pi-nicho
#SBATCH --mail-type=ARRAY_TASKS,FAIL
#SBATCH --mail-user=daltonm@uchicago.edu

#------------params that may change for each dataset/experiment----------------------#
date=2023_08_09
exp1=moth
exp2=moth_free
sess=(1 2)
video_neural_session_alignment=matched #(matched, all_in_one_neural_recording)
touchscreen=False

#----------------------params that rarely change-------------------------------------#
marms=JLTY
fps=30
ncams=4
cams=(1 2 3 4)
vid_transpose=(2 2 1 1)  # number of arguments MUST match ncams 
analog_signals_fps=(200 $fps)

marms_ephys=JL
swap_ab=no
prb_path=/project/nicho/data/marmosets/prbfiles/JL_01.prb
metadata_path=/project/nicho/data/marmosets/metadata_yml_files/JL_complete_metadata.yml

#-----------paths to input and output data, and to processing scripts----------------#
vid_dir=/project/nicho/data/marmosets/kinematics_videos
ephys_dir=/project/nicho/data/marmosets/electrophys_data_for_processing
video_processing=/project/nicho/projects/marmosets/code_database/data_processing/kinematics/video_processing
neural_processing=/project/nicho/projects/marmosets/code_database/data_processing/neural

touchscreen_path=BLANK

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /beagle3/nicho/environments/nwb_and_neuroconv

#---------------------Execute the programs-------------------------------------------#
python $video_processing/process_analog_signals_for_episode_times.py \
-v $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d $date -e $exp1 -e2 $exp2 \
-t $touchscreen -tp $touchscreen_path -np $neural_processing -meta $metadata_path \
-prb $prb_path -ab $swap_ab -al $video_neural_session_alignment -f ${analog_signals_fps[@]}

python $neural_processing/neural_dropout_first_pass.py -k $vid_dir -ep $ephys_dir -m $marms \
-me $marms_ephys -d $date -e $exp1 -e2 $exp2
