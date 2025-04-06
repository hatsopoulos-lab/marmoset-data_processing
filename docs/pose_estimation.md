# Pose Estimation Guide

How to run pose estimation using DLC and Anipose on midway3 computing cluster.

>Throughout this guide, you will want to use the following setup on midway3

        sinteractive --account=pi-nicho --mem=96G --partition=beagle3 --time=6:00:00 --gres=gpu:1
        module load pytorch/1.10
        source activate /beagle3/nicho/environments/anipose-dlc-pytorch

You will want to use the most up-to-date model, which is located at `/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2024-10-27`.

### Move file to the correct locations

If this is the first time working with this set of marmosets or experiment:

        mkdir -p /project/nicho/data/marmosets/kinematics_videos/EXP/MARMS/
        cp /project/nicho/projects/marmosets/code_database/data_processing/subject_specific_scripts/TEMPLATES/TEMPLATE_config.toml /project/nicho/data/marmosets/kinematics_videos/EXP/MARMS/config.toml

Copy recording session videos from cds3 to project

        cp -r /cds3/nicho/data/marmosets/kinematics_videos/EXP/MARMS/YYYY_MM_DD /project/nicho/data/marmosets/kinematics_videos/EXP/MARMS/


### Start by identifying the video events that can contain the studied behavior

This can be done most simply by visual inspection of the videos. Keep a record of this, preferably in a google sheet. You may want to add more information and refer to this in later steps. Move all video events without the studied behavior to a separate folder within the directory, something like `no_behavior_avi_videos`.

### Prepare DLC for new video data and extract frames

>If you think the data is similar to data that is already represented in the training set, you can skip this step.

Copy [TEMPLATE_extract_frames_for_dlc_labeling_or_refinement.py](/subject_specific_scripts/TEMPLATES/python/TEMPLATE_extract_frames_for_dlc_labeling_or_refinement.py) into your `subject_specific_files/MARM/` directory and use the existing dict entries to create a new entry for your subject. Start with ~5-10 frames from each camera, choosing a different video event for each camera. The start and stop fractions should be the fraction of the way thru the video that interesting behavior starts and ends. The more accurate you are with this, the more useful your labels will be. The "mode" entry should be set to "original". Now run this in an sinteractive job, either in the terminal or in spyder, to add videos to the dlc project and extract frames for labeling.

### Label frames

>If you didn't extract any new frames or refined labels, skip this.

Follow the deeplabcut guide for initial labeling. Most of the steps are simplest if you also have the Files app open on ThinLinc and drag the files you need into Napari. For example, for new labeling you can drag the config.yaml file in then drag fodler of images in. If you are refining labels, you should: 

1. Delete the existing config.yaml from the napari window.
2. Drag in the folder to refine (will have a machineLabels file inside). 
3. Make sure the machinelabels file is highlighted and save the labels. This will merge the machinelabels with the existing CollectedData labels, or create that file. Then you should adjust the labels as you go and save again at the end of label adjustments. Editing the machinelabels first, then merging, will often cause a bug that kills the napari labeling GUI (this may be fixed in a future DLC update). 

> Tips: 
> - If the landmark is easy to see, label it.
> - If the landmark is difficult to label (for example, fully or partially occluded), you should skip it if you are confident that at least two other cameras will be able to see it well or if you don't need the data of that frame. You should label it if it's one of the two best views you have and you need the data in that posture.  

### Train the network

>If you didn't label or refine frames, skip this step.

Copy and edit the parameters in [TEMPLATE_train_full_marmoset_dlc_model.sbatch](/subject_specific_scripts/TEMPLATES/sbatch/TEMPLATE_train_full_marmoset_dlc_model.sbatch). Make sure `init_weights` is set to the most recent iteration and snapshot of the model. Leave `batch_size=16`, because this is the largest batch size that works with the GPU memory and you want a larger batch size to improve generalization. Make sure you have updated the iteration in the config.yaml file of the project, then run this with sbatch. <br>

You can also edit parameters in `/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2024-10-27/pytorch_config_template.yaml` (which has a copy at [pytorch_config_template.yaml](/project/nicho/projects/marmosets/code_database/data_processing/subject_specific_scripts/TEMPLATES/pytorch_config_template.yaml)), although this is not suggested. The `resize` param is critical, as it determines the "receptive field" for nodes in the CNN. Basically, a smaller image allows CNN nodes to see more of the image, while a larger image confines the nodes more locally. A 640x480 image works well for accurate labeling that generalizes well (and can distinguish right side from left side markers).

### Analyze videos with Anipose

Copy and edit [TEMPLATE_anipose_marmosets_job_submission.sbatch](/subject_specific_scripts/TEMPLATES/sbatch/TEMPLATE_anipose_marmosets_job_submission.sbatch). Then run as an sbatch job.

### Inspect the videos

Do a first-pass assessment of labeling quality by looking thru the videos in videos-2d-proj. If the videos are poorly labeled, you can use other video files to understand if this happened during triangulation or just do to poor coverage in the training set of the network. If triangulation is the problem, take a look at the calibration error (should be <10). You may have to inspect calibration images using [TEMPLATE_check_for_bad_calibration_images.py](/subject_specific_scripts/TEMPLATES/python/TEMPLATE_check_for_bad_calibration_images.py). If more labels are need, go back and label more frames from scratch, and re-train. Once the videos are looking good and have only occasional errors, move on.

### Refine pose and create diagnostic videos

1. Use [TEMPLATE_refine_pose.py](/subject_specific_scripts/TEMPLATES/python/TEMPLATE_refine_pose.py) to do post-processing of pose, generate diagnostic plots, and refined pose files. *NOTE: this script could use some attention put towards simplifying and cleaning it up.*
2. Use [TEMPLATE_make_3D_videos.sbatch](/subject_specific_scripts/TEMPLATES/sbatch/TEMPLATE_make_3D_videos.sbatch) to create projected videos from the refined pose. These videos vizualize the final, pose-processed pose data so you can decide whther it is good enough to be saved in the to `_processed.nwb` file for this dataset.
3. Inspect these videos carefully, along with the diagnostic plots created in step 1. Note the total frame count and frame numbers at the start/end of outlier segments that will need to be refined in the google sheet you are using to track this dataset. Also create columns that comptue the start/stop fraction of the video, which will guide the dict entries when you go back to frame extraction (see *Prepare DLC for new video data and extract frames* above).
4. Store these results in a new folder as a record of this iteration of pose estimation. From within the date folder, for example:

        mkdir RESULTS_full_marmoset_model_iter1
        mv videos-2d-proj/ videos-labeled-filtered/ pose-* config.toml scorer_info.txt autoencoder.pickle RESULTS_full_marmoset_model_iter1/ 

5. If you plan to run anipose again after label refinement, remove the other video files:

        rm -r videos-*

### Refine labels

Go back to *Prepare DLC for new video data and extract frames* and repeat. Try out doing this with mode='original' for a small subset and mode='outlier' for a subset and choose whichever option you find easiest.





