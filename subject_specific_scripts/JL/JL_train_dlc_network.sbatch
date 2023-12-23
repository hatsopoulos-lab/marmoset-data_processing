#!/bin/bash
#SBATCH --job-name=train_dlc_marms
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/train_dlc_marms.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/train_dlc_marms.err
#SBATCH --time=36:00:00
#SBATCH --partition=beagle3
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#---------------------------------------------Params that may change often----------------#
dates_to_copy=()
ncams=5
nlabels=3
dlc_scorer=Dalton
experiment=apparatus

maxiters=5000000
init_weights=/project/nicho/projects/marmosets/dlc_project_files/simple_marmoset_model-Dalton-2023-04-28/dlc-models/iteration-9/simple_marmoset_modelApr28-trainset95shuffle1/train/snapshot-1060000
#init_weights=None
overwrite_trainset=False

#---------------------------------------------Params unlikely to change-------------------#
param_set=marmoset
if [[ "$experiment" == *"free"* ]] || [[ "$experiment" == *"enclosure"* ]]; then
    dlc_path=/project/nicho/projects/marmosets/undergraduate_dlc_projects/spontaneous_behavior-undergrads-2023-01-13
else
    dlc_path=/project/nicho/projects/marmosets/dlc_project_files/simple_marmoset_model-Dalton-2023-04-28
fi

#---------------------------------------------Load the appropriate applications-----------#
module load cudnn/11.2
module load python/anaconda-2022.05 
source activate /project/nicho/environments/dlc

#---------------------------------------------Execute the programs------------------------#
#python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/copy_origin_and_axis_labels_to_unmarked_labeled_frames.py \
#-dp $dlc_path -d ${dates_to_copy[@]} -c $ncams -l $nlabels -s $dlc_scorer

python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/train_dlc.py \
-d $dlc_path -m $maxiters -i $init_weights -o $overwrite_trainset

