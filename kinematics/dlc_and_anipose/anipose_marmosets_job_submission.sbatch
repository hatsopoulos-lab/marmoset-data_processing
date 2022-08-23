#!/bin/bash
#SBATCH --job-name=anipose_marms
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/anipose_marms.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/anipose_marms.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --mem=40G
#SBATCH --account=pi-nicho

#---------------------------------------------Params that may change often----------------#
experiment=test
marms=TYJL
train_frac=None
snap_idx=-1
dlc_iter=None
extra_vars=(False True) # (only_3D label_videos)

#---------------------------------------------Params unlikely to change-------------------#
param_set=marmoset
if [[ "$experiment" == *"free"* ]] || [[ "$experiment" == *"enclosure"* ]]; then
    dlc_path=/project/nicho/projects/marmosets/dlc_project_files/FREE_BEHAVIOR_PROJECT_NAME
else; then
    dlc_path=/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26
fi
anipose_path=/project/nicho/data/marmosets/kinematics_videos/$experiment/$marms

#---------------------------------------------Load the appropriate applications-----------#
module load python/anaconda-2021.05 
module load cudnn/11.2
source activate /project/nicho/environments/dlc

#---------------------------------------------Execute the programs------------------------#
python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/run_anipose.py \
-d $dlc_path -a $anipose_path -i $dlc_iter -f $train_frac -s $snap_idx -p $param_set -v ${extra_vars[@]}