# Marmoset Data Processing Pipeline

This code is intended for internal use within the Hatsopoulos Lab, and is designed to operate primarily on the RCC midway3 computing cluster.

## Contents
  1. **batch_job_files**: contains primarily sbatch scripts and some .py files tailored to individual marmosets or projects. These use code contained in the other three folders.
  2. **kinematics**:      contains code for processing behavioral data videos and compute pose estimation with DLC+Anipose, including custom tools written for specific lab needs.
  3. **neural**:          contains code for processing neural data and additing spike-sorted data to nwb files.
  4. **nwb_tools**:       contains nwb-related tools, including guides written in jupyter notebook for adding data to NWB files and accessing the data in these files.

## Overview of Pipeline - Click on links for further detail
### Initial processing of raw data (insert link to video_processing docs)
  1. The process begins with extracted jpg files for a particular date in /scratch/CNETID/kinematics_jpgs/GOAL_DIRECTED_EXP_NAME/YYYY_MM_DD and /scratch/CNETID/kinematics_jpgs/FREE_EXP_NAME/YYYY_MM_DD, along with the folder containing the corresponding neural data in /project/nicho/data/marmosets/electrophys_data_for_processing.
  2. Videos are produced from jpgs for both the goal-directed and free behavior experiments (aka apparatus and enclosure), then video frames are aligned to neural data using the analog signals stored in the .ns6 neural data file. An NWB file is generated containing raw neural data from the .ns6 file, unsorted threshold crossing timestamps from the .nev file, aligned video frame timestamps, and potential periods of neural dropout. The NWB file is stored in the electrophysiology data folder with the extension '_acquisition.nwb'.

### Computing pose with DLC+Anipose (insert link to dlc docs)
  1. The most recent version of the desired DLC project can be applied immediately as described below, or updated with new labeled frames corresponding to fresh data.
  2. DLC+Anipose is applied to video data to track movements of the marmoset in 2D pixel space, then triangulated to 3D coordinates.
  3. Videos should be examined and the DLC network refined as necessary.
  4. Once DLC network produces acceptable pose estimation, the 3D pose is processed and prepared for the NWB file.
  5. If all pose estimation data is satisfactory it is added to the NWB file and associated with the corresponding timestamps.

### Spike-sorting (insert link neural to  docs)
  1. Run .ns6 file thru spike-sorting notebook (at present, this step must be run on a local machine and Paul is the point-person).
  2. Place spike-sorting output in scratch space on midway3 and manually curate the sorting using phy2.
  3. Add curated spike-sorting data to NWB file.

### Now you are ready to access and analyze the data!
