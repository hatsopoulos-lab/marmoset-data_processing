#!/bin/bash
#SBATCH --job-name=JL_anipose
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/JL_anipose_0804_fullModel_%a.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/JL_anipose_0804_fullModel_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=beagle3
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --array=0-4 #0
#SBATCH --mem=96G
#SBATCH --account=pi-nicho

#---------------------------------------------Params that may change often----------------#
experiment=moth
marms=JLTY
dates=(2023_08_04)
train_frac=None
snap_idx=-1
dlc_iter=None
extra_vars=(False True) # (only_3D label_videos)
copy_up_to=None #pose-2d-viterbi_and_autoencoder #pose-2d-unfiltered #None

ncams=5 # maybe change to 2 for unilateral marking
vid_transpose=(-1 -1 -1 -1 -1)  # number of arguments MUST match ncams 

#---------------------------------------------Params unlikely to change-------------------#
param_set=marmoset
if [[ "$experiment" == *"free"* ]] || [[ "$experiment" == *"enclosure"* ]]; then
    dlc_path=/project/nicho/projects/marmosets/dlc_project_files/FREE_BEHAVIOR_PROJECT_NAME
else
    dlc_path=/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2024-10-27
    #dlc_path=/project/nicho/projects/marmosets/dlc_project_files/simple_5cams_marmoset_model-Dalton-2024-06-27
    #dlc_path=/project/nicho/projects/marmosets/dlc_project_files/simple_marmoset_model-Dalton-2023-04-28 
fi
anipose_path=/project/nicho/data/marmosets/kinematics_videos/$experiment/$marms #/$dates/$videotype

#---------------------------------------------Load the appropriate applications-----------#
#module load cudnn/11.2
#module load python/anaconda-2023.09 
#source activate /project/nicho/environments/dlc
module load pytorch/1.10
source activate /beagle3/nicho/environments/anipose-dlc-pytorch

#---------------------------------------------Execute the programs------------------------#
#python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/check_for_and_create_calibration_videos.py \
#-a $anipose_path -t ${vid_transpose[@]} -n $ncams

pytorch=True
python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/run_anipose.py \
-d $dlc_path -a $anipose_path -i $dlc_iter -f $train_frac -s $snap_idx -p $param_set -v ${extra_vars[@]} \
-dt ${dates[@]} -n $ncams -c $copy_up_to -pt $pytorch

module unload pytorch/1.10
module load cudnn/11.2
module load python/anaconda-2023.09 
source activate /project/nicho/environments/dlc

pytorch=False
python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/run_anipose.py \
-d $dlc_path -a $anipose_path -i $dlc_iter -f $train_frac -s $snap_idx -p $param_set -v ${extra_vars[@]} \
-dt ${dates[@]} -n $ncams -c $copy_up_to -pt $pytorch
