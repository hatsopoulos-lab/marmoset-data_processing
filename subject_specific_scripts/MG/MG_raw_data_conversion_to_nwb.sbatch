#!/bin/bash
#SBATCH --job-name=MG_nwb_conv
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/MG_nwb_conv_20230416.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/MG_nwb_conv_20230416.err
#SBATCH --time=2:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#------------params that may change for each dataset/experiment----------------------#
dates=(2023_04_16)
exp1=moths
exp2=free
sess=(1)
video_neural_session_alignment=all_in_one_neural_recording #(matched, all_in_one_neural_recording)
touchscreen=False


#----------------------params that rarely change-------------------------------------#
marms=HMMG
marms_ephys=MG
swap_ab=no
prb_path=/project/nicho/data/marmosets/prbfiles/MG_01.prb
metadata_path=/project/nicho/data/marmosets/metadata_yml_files/MG_complete_metadata.yml
analog_signals_fps=(200 60)

#-----------paths to input and output data, and to processing scripts----------------#
vid_dir=/project/nicho/data/marmosets/kinematics_videos
ephys_dir=/project/nicho/data/marmosets/electrophys_data_for_processing
video_processing=/project/nicho/projects/marmosets/code_database/data_processing/kinematics/video_processing
neural_processing=/project/nicho/projects/marmosets/code_database/data_processing/neural

touchscreen_path=BLANK

#------------Load the appropriate applications---------------------------------------#
module load python/anaconda-2022.05 
source activate /project/nicho/environments/spyder

#---------------------Execute the programs-------------------------------------------#

source activate /beagle3/nicho/environments/nwb_and_neuroconv

for date in ${dates[@]}
do
    python $video_processing/process_analog_signals_for_episode_times.py  \
    -v $vid_dir -ep $ephys_dir -m $marms -me $marms_ephys -d $date -e $exp1 -e2 $exp2 \
    -t $touchscreen -tp $touchscreen_path -np $neural_processing -meta $metadata_path \
    -prb $prb_path -ab $swap_ab -al $video_neural_session_alignment -f ${analog_signals_fps[@]}

    python $neural_processing/neural_dropout_first_pass.py -k $vid_dir -ep $ephys_dir -m $marms \
    -me $marms_ephys -d $date -e $exp1 -e2 $exp2
done