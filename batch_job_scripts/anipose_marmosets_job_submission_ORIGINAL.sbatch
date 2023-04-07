#!/bin/bash
#SBATCH --job-name=anipose_marms
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/anipose_marms.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/anipose_marms.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --account=pi-nicho

experiment='test'

dates='all'
train_frac='None'
snap_idx='-1'
dlc_iter='None'
extra_vars='False False' # 'add_event_only only3D'

param_set='marmoset'
dlc_path='/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26'
anipose_path="/project/nicho/data/marmosets/kinematics_videos/$experiment"

# Load the appropriate applications
module load python/anaconda-2021.05 
source activate /project/nicho/environments/dlc

# Execute the programs

python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/run_anipose.py \
-d $dlc_path -a $anipose_path -i $dlc_iter -f $train_frac -s $snap_idx -p $param_set -dt $dates -v $extra_vars