#!/bin/bash
#SBATCH --job-name=train_dlc_marms
#SBATCH --output=/project/nicho/projects/marmosets/job_log_files/dlc_pytorch_marms.out
#SBATCH --error=/project/nicho/projects/marmosets/job_log_files/dlc_pytorch_marms.err
#SBATCH --time=20:00:00
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

batch_size=16
maxiters=5000000
init_weights=/project/nicho/projects/marmosets/dlc_project_files/simple_5cams_marmoset_model-Dalton-2024-06-27/dlc-models-pytorch/iteration-5/simple_5cams_marmoset_modelJun27-trainset95shuffle1/train/snapshot-100.pt
#init_weights=None
overwrite_trainset=False

#---------------------------------------------Params unlikely to change-------------------#
param_set=marmoset
if [[ "$experiment" == *"free"* ]] || [[ "$experiment" == *"enclosure"* ]]; then
    dlc_path=/project/nicho/projects/marmosets/undergraduate_dlc_projects/spontaneous_behavior-undergrads-2023-01-13
else
    dlc_path=/project/nicho/projects/marmosets/dlc_project_files/simple_5cams_marmoset_model-Dalton-2024-06-27
    template_path=/project/nicho/projects/marmosets/dlc_project_files/simple_5cams_marmoset_model-Dalton-2024-06-27/pytorch_config_template.yaml 
    #simple_marmoset_model-Dalton-2023-04-28
fi

#---------------------------------------------Load the appropriate applications-----------#
#module load cudnn/11.2
#module load python/anaconda-2023.09 
#source activate /project/nicho/environments/dlc
module load pytorch/1.10 
source activate /beagle3/nicho/environments/dlc-pytorch

#---------------------------------------------Execute the programs------------------------#
#python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/copy_origin_and_axis_labels_to_unmarked_labeled_frames.py \
#-dp $dlc_path -d ${dates_to_copy[@]} -c $ncams -l $nlabels -s $dlc_scorer

python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/print_pytorch_cuda_cudnn.py

python /project/nicho/projects/marmosets/code_database/data_processing/kinematics/dlc_and_anipose/train_dlc.py \
-d $dlc_path -m $maxiters -i $init_weights -o $overwrite_trainset -b $batch_size -t $template_path

