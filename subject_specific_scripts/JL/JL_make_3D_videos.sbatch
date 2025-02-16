#!/bin/bash
#SBATCH --job-name=make_3D_videos
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/JL_3D_videos_%a.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/JL_3D_videos_%a.err
#SBATCH --time=1:00:00
#SBATCH --partition=beagle3 #caslake
#SBATCH --gres=gpu:1
#SBATCH --array=0-25 # make sure this matches the total number of events
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --account=pi-nicho

#---------------------------------------------Params that may change often----------------#

date=2023_08_04
anipose_path=/project/nicho/data/marmosets/kinematics_videos/moth/JLTY
cam=2
post_processed=True
overwrite=True
end_cut_time=10  #the time (in seconds) at the end of every video that should be cut 

#---------------------------------------------Params unlikely to change-------------------#

#---------------------------------------------Load the appropriate applications-----------#
module load ffmpeg/5.1
module load python/anaconda-2023.09 
source activate /beagle3/nicho/environments/mayavi

#---------------------------------------------Execute the programs------------------------#

python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/make_3d_label_videos.py \
-d $date -o $overwrite -p $post_processed -a $anipose_path -c $cam -e $end_cut_time 

