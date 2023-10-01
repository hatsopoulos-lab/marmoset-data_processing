#!/bin/bash
#SBATCH --job-name=MG_calibrate
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/MG_calibrate.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/MG_calibrate.err
#SBATCH --time=06:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#---------------------------------------------Params that may change often----------------#
experiment=free
marms=HMMG
dates=(2023_03_17 2023_04_16 2023_04_17)

ncams=4
vid_transpose=(2 2 1 1) # number of arguments MUST match ncams 

#---------------------------------------------Params unlikely to change-------------------#

anipose_path=/project/nicho/data/marmosets/kinematics_videos/$experiment/$marms

#---------------------------------------------Load the appropriate applications-----------#
module load python/anaconda-2022.05 
source activate /project/nicho/environments/dlc

#---------------------------------------------Execute the programs------------------------#
python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/check_for_and_create_calibration_videos.py \
-a $anipose_path -t ${vid_transpose[@]} -n $ncams

python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/calibrate_for_anipose.py \
-a $anipose_path -dt ${dates[@]}