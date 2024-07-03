#!/bin/bash
#SBATCH --job-name=copy_axes
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/copy_axes.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/copy_axes.err
#SBATCH --time=2:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --account=pi-nicho

#---------------------------------------------Params that may change often----------------#
dates_to_copy=(2023_08_03)
ncams=5
nlabels=3
dlc_scorer=Dalton
experiment=apparatus

#---------------------------------------------Params unlikely to change-------------------#
param_set=marmoset
if [[ "$experiment" == *"free"* ]] || [[ "$experiment" == *"enclosure"* ]]; then
    dlc_path=/project/nicho/projects/marmosets/undergraduate_dlc_projects/spontaneous_behavior-undergrads-2023-01-13
else
    dlc_path=/project/nicho/projects/marmosets/dlc_project_files/simple_marmoset_model-Dalton-2023-04-28
fi

#---------------------------------------------Load the appropriate applications-----------#
module load python/anaconda-2022.05 
source activate /project/nicho/environments/dlc

#---------------------------------------------Execute the programs------------------------#
python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/copy_origin_and_axis_labels_to_unmarked_labeled_frames.py \
-dp $dlc_path -d ${dates_to_copy[@]} -c $ncams -l $nlabels -s $dlc_scorer