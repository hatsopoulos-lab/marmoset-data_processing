#!/bin/bash
#SBATCH --job-name=spykingcircus
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/spykingcircus.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/spykingcircus.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --account=pi-nicho
#SBATCH --mail-user=daltonm@uchicago.edu
#SBATCH --mail-type=ALL
#---------------------------------------------Params that may change often----------------#


#---------------------------------------------Params unlikely to change-------------------#


#---------------------------------------------Load the appropriate applications-----------#
module load matlab/2022b
module load python/anaconda-2022.05 
source activate /project/nicho/environments/spikes

#---------------------------------------------Execute the programs------------------------#
python /project/nicho/projects/marmosets/code_database/data_processing/neural/process_data_for_spikesorting_and_run_spykingcircus.py