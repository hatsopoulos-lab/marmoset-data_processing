#!/bin/bash
#SBATCH --job-name=MG_anipose
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/MG_anipose_%a.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/MG_anipose_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=beagle3
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --array=0-4 #0
#SBATCH --mem=96G
#SBATCH --account=pi-nicho

#---------------------------------------------Params that may change often----------------#
experiment=moths
marms=HMMG
dates=(2023_04_16)
train_frac=None
snap_idx=-1
dlc_iter=None
extra_vars=(False True) # (only_3D label_videos)

ncams=5
vid_transpose=(-1 -1 -1 -1 -1)  # number of arguments MUST match ncams 

#---------------------------------------------Params unlikely to change-------------------#
param_set=marmoset
if [[ "$experiment" == *"free"* ]] || [[ "$experiment" == *"enclosure"* ]]; then
    dlc_path=/project/nicho/projects/marmosets/dlc_project_files/FREE_BEHAVIOR_PROJECT_NAME
else
    dlc_path=/project/nicho/projects/marmosets/dlc_project_files/simple_marmoset_model-Dalton-2023-04-28 
    #dlc_path=/project/nicho/projects/marmosets/dlc_project_files/simple_MG_model-Dalton-2023-04-28
fi
anipose_path=/project/nicho/data/marmosets/kinematics_videos/$experiment/$marms

#---------------------------------------------Load the appropriate applications-----------#
module load cudnn/11.2
module load python/anaconda-2022.05 
source activate /project/nicho/environments/dlc

#---------------------------------------------Execute the programs------------------------#
#python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/check_for_and_create_calibration_videos.py \
#-a $anipose_path -t ${vid_transpose[@]} -n $ncams

python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/run_anipose.py \
-d $dlc_path -a $anipose_path -i $dlc_iter -f $train_frac -s $snap_idx -p $param_set -v ${extra_vars[@]} -dt ${dates[@]} -n $ncams
