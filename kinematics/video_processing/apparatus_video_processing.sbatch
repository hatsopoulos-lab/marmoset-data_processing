#!/bin/bash
#SBATCH --job-name=filter_videos_preDLC
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/filter_videos_preDLC_%a.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/filter_videos_preDLC_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=caslake
#SBATCH --array=1-20
#SBATCH --ntasks=4
#SBATCH --mem=4G
#SBATCH --account=pi-nicho

dates='2022_06_17'
marms='TYJL'
jpg_dir='/scratch/midway3/daltonm/kinematics_jpgs'
vid_dir='/project/nicho/data/marmosets/kinematics_videos'

# apparatus experiment params
app_exp='test'
app_sess='1'
app_fps='200'
app_ncams='5'
app_vid_transpose='-1 -1 -1 -1 -1' # number of arguments MUST match app_ncams

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
#/project/nicho/projects/marmoset/code_database/processing/apparatus_jpg2avi.sh 'TYJL' '2022_04_12' 'test' '3' '200'
#python /project/nicho/projects/marmoset/code_database/processing/apply_clahe_filter_to_videos.py -d /project/nicho/data/marmosets/kinematics_videos/test/TYJL/ -n 1 -p 28 

#/project/nicho/projects/marmoset/code_database/processing/free_jpg2avi.sh 'TYJL' '2022_04_12' 'test_free' '3' '60'
#python /project/nicho/projects/marmoset/code_database/processing/apply_clahe_filter_to_videos.py -d /project/nicho/data/marmosets/kinematics_videos/test_free/TYJL/ -n 1 -p 28
#python /project/nicho/projects/marmoset/code_database/processing/recompress_filtered_avi.py      -d /project/nicho/data/marmosets/kinematics_videos/test_free/TYJL/ -n 1 -p 28  

###################################################################

python /project/nicho/projects/marmosets/code_database/processing/jpg2avi.py -j $jpg_dir -v $vid_dir -m $marms -d $dates -e $app_exp -s $app_sess -f $app_fps -n $app_ncams -t $app_vid_transpose
python /project/nicho/projects/marmosets/code_database/processing/apply_clahe_filter_to_videos.py -i $vid_dir/$app_exp/$marms/ -d $dates
echo 'reached beginning of recompression'
python /project/nicho/projects/marmosets/code_database/processing/recompress_filtered_avi.py      -i $vid_dir/$app_exp/$marms/ -d $dates  

python /project/nicho/projects/marmosets/code_database/processing/jpg2avi.py -j $jpg_dir -v $vid_dir -m $marms -d $dates -e $free_exp -s $free_sess -f $free_fps -n $free_ncams -t $free_vid_transpose   
python /project/nicho/projects/marmosets/code_database/processing/apply_clahe_filter_to_videos.py -i $vid_dir/$free_exp/$marms/ -d $dates
python /project/nicho/projects/marmosets/code_database/processing/recompress_filtered_avi.py      -i $vid_dir/$free_exp/$marms/ -d $dates
