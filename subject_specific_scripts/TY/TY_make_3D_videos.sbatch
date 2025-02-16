#!/bin/bash
#SBATCH --job-name=make_3D_videos
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/TY_3D_videos_0211_%a.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/TY_3D_videos_0211_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=caslake
#SBATCH --array=0-77 # make sure this matches the total number of events
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#---------------------------------------------Params that may change often----------------#

date=2021_02_11
anipose_path=/project/nicho/data/marmosets/kinematics_videos/moth/TYJL
cam=1
post_processed=True
overwrite=True

#---------------------------------------------Params unlikely to change-------------------#

#---------------------------------------------Load the appropriate applications-----------#
module load ffmpeg/5.1
module load python/anaconda-2023.09 
source activate /beagle3/nicho/environments/mayavi

#---------------------------------------------Execute the programs------------------------#

python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/make_3d_label_videos.py \
-d $date -o $overwrite -p $post_processed -a $anipose_path -c $cam

