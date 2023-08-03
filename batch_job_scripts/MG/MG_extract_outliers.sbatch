#!/bin/bash
#SBATCH --job-name=MG_extract_outliers
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/MG_extract_outliers.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/MG_extract_outliers.err
#SBATCH --time=24:00:00
#SBATCH --partition=beagle3
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --account=pi-nicho

module load cudnn/11.2
module load python/anaconda-2022.05 
source activate /project/nicho/environments/dlc

python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/extract_outlier_frames_for_dlc_refinement.py