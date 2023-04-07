#!/bin/bash
#SBATCH --job-name=filter_videos_preDLC
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/filter_videos_preDLC.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/filter_videos_preDLC.err
#SBATCH --time=12:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=500
#SBATCH --account=pi-nicho

dates='2022_06_17'
marms='TYJL'

# apparatus experiment params
app_exp='test'
app_sess='1'
app_fps='200'
app_ncams='5'
app_vid_transpose='-1'

# free experiment params
free_exp='test_free'
free_sess='1' 
free_fps='60'
free_ncams='3'
free_vid_transpose='2'
 
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

python /project/nicho/projects/marmosets/code_database/processing/jpg2avi.py -m $marms -d $dates -e $app_exp -s $app_sess -f $app_fps -n $app_ncams -t $app_vid_transpose
python /project/nicho/projects/marmosets/code_database/processing/apply_clahe_filter_to_videos.py -d /project/nicho/data/marmosets/kinematics_videos/$app_exp/$marms/ -n 1 -p 96 
echo 'reached beginning of recompression'
python /project/nicho/projects/marmosets/code_database/processing/recompress_filtered_avi.py      -d /project/nicho/data/marmosets/kinematics_videos/$app_exp/$marms/ -n 1 -p 0  

python /project/nicho/projects/marmosets/code_database/processing/jpg2avi.py -m $marms -d $dates -e $free_exp -s $free_sess -f $free_fps -n $free_ncams -t $free_vid_transpose   
python /project/nicho/projects/marmosets/code_database/processing/apply_clahe_filter_to_videos.py -d /project/nicho/data/marmosets/kinematics_videos/$free_exp/$marms/ -n 1 -p 96
python /project/nicho/projects/marmosets/code_database/processing/recompress_filtered_avi.py      -d /project/nicho/data/marmosets/kinematics_videos/$free_exp/$marms/ -n 1 -p 96  

