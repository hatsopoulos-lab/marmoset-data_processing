#!/bin/bash
#SBATCH --job-name=filter_videos_preDLC
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/filter_videos_preDLC_%a.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/filter_videos_preDLC_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=caslake
#SBATCH --array=1-20
#SBATCH --ntasks=8
#SBATCH --mem=4G
#SBATCH --account=pi-nicho

dates='2022_06_17'
marms='TYJL'
jpg_dir='/scratch/midway3/daltonm/kinematics_jpgs'
vid_dir='/project/nicho/data/marmosets/kinematics_videos'

# free experiment params
free_exp='test_free'
free_sess='1' 
free_fps='60'
free_ncams='3'
free_vid_transpose='2 2 2' # number of arguments MUST match free_ncams
 
# Load the appropriate applications
module load python/anaconda-2021.05 
source activate /project/nicho/environments/dlc

# Execute the program
python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/jpg2avi.py -j $jpg_dir -v $vid_dir -m $marms -d $dates -e $free_exp -s $free_sess -f $free_fps -n $free_ncams -t $free_vid_transpose   
python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/apply_clahe_filter_to_videos.py -i $vid_dir/$free_exp/$marms/ -d $dates
python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/recompress_filtered_avi.py      -i $vid_dir/$free_exp/$marms/ -d $dates
