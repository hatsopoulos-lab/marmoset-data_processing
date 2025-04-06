# Video Processing Guide

How to process marmoset video data on midway3 computing cluster. See last section for troubleshooting tips.

### Code used in this guide

[TEMPLATE_apparatus_video_processing.sbatch](/subject_specific_scripts/TEMPLATES/sbatch/TEMPLATE_apparatus_video_processing.sbatch)<br>
[TEMPLATE_enclosure_video_processing.sbatch](/subject_specific_scripts/TEMPLATES/sbatch/TEMPLATE_enclosure_video_processing.sbatch)<br>
[TEMPLATE_sleep_video_processing.sbatch](/subject_specific_scripts/TEMPLATES/sbatch/TEMPLATE_sleep_video_processing.sbatch)<br>
[check_for_episode_splits_and_adjust_image_filenames.py](/kinematics/video_processing/check_for_episode_splits_and_adjust_image_filenames.py)<br>
[jpg2avi.py](/kinematics/video_processing/jpg2avi.py)<br>
[process_analog_signals_for_episode_times.py](/kinematics/video_processing/process_analog_signals_for_episode_times.py)<br> 
[neural_dropout_first_pass.py](/neural/neural_dropout_first_pass.py)<br>
[TEMPLATE_validate_nwb_files.py](/subject_specific_scripts/TEMPLATES/python/TEMPLATE_validate_nwb_files.py)

### Check the data 
1.	Access midway3 via ThinLinc or ssh
2.	Check that jpg files are extracted to the correct locations in your scratch space:

		Goal-directed: /SCRATCH_DIR/kinematics_jpgs/GOAL_DIRECTED_EXP_NAME/YYYY_MM_DD
        Spontaneous:   /SCRATCH_DIR/kinematics_jpgs/FREE_EXP_NAME/YYYY_MM_DD 
            
        If necessary, extract archived files with: tar -xf YYYY_MM_DD.tar
        
3.	Check that neural data folder containing ns6 and nev files is stored at: 
    
    	/DATA_DIR/electrophys_data_for_processing/

### Prepare and run sbatch jobs
1.	If this is the first time processing data for this marmoset, 
[prepare metadata and prb files, and create subject-specific job scripts.](/docs/prepare_for_new_subject.md)

2.	Open files below in your text editor of choice:

        /DATA_PROCESSING_DIR/subject_specific_files/MARM/MARM_apparatus_video_processing.sbatch
        /DATA_PROCESSING_DIR/subject_specific_files/MARM/MARM_enclosure_video_processing.sbatch
	
    (For sleep experiments, use `/DATA_PROCESSING_DIR/subject_specific_files/MARM/MARM_sleep_video_processing.sbatch`)


3.	Edit the SBATCH job parameters as necessary
4.	Edit the parameters in the first section, denoted: 
`#------------params that may change for each dataset/experiment---------------#`
5.	Check that parameters in the next section are correct: 
`#----------------------params that rarely change------------------------------#`
6.	Run the jobs:

		sbatch /DATA_PROCESSING_DIR/subject_specific_files/MARM/MARM_apparatus_video_processing.sbatch
        sbatch /DATA_PROCESSING_DIR/subject_specific_files/MARM/MARM_enclosure_video_processing.sbatch

7.	Check status of jobs with:
		
        squeue --me
        vi /path/to/job_log_files/JOB_LABEL_jobNum.out
        vi /path/to/job_log_files/JOB_LABEL_jobNum.err
        
### Check data after jobs end
1.	Look through the video files in `/DATA_DIR/kinematics_videos/EXP_NAME/MARM/YYYY_MM_DD/avi_videos`. The file count should be equal to N_cameras * N_video_events. Check that all videos have reasonable data sizes (shouldn't be 0 or 1 kB).

2.	Check that the acquisition.nwb file was created correctly by editing and running [TEMPLATE_validate_nwb_files.py](/subject_specific_scripts/TEMPLATES/python/TEMPLATE_validate_nwb_files.py) in a Spyder instance. This script also provides a basic introduction to extracting the information within an NWB file.

    Alternatively, you can explore it with nwbwidget:

		sinteractive --partition=caslake --mem=64G --time=5:00:00 --account=pi-nicho
		module load python/anaconda-2023.09
		source activate /project/nicho/environments/spyder
		cd /DATA_PROCESSING_DIR/nwb_tools
        jupyter notebook
        
    Open `simple_nwbwidget.ipynb` and use it to look thru neural data, stored video 
    timestamp data, and metadata.
    
3.	If you are unsure whether the data is complete/accurate, you should inspect some intermediate 
processing files located in `/metadata_from_kinematics_processing` and `/drop_records`. There should
be two files located here, which can be opened in any python IDE (iPython, spyder, etc):

		with open('/DATA_DIR/kinematics_videos/EXP_NAME/MARM/YYYY_MM_DD/metadata_from_kinematics_processing/YYYYMMDD_experiment_event_and_frame_time_info.pkl', 'rb') as f:
        	metadata = dill.load(f)

### Clean up intermediate files

Once you are certain the videos were created correctly, delete the jpg files and archived folders from your scratch space. 

>If you are not completely sure the videos are completely converted, make sure you have 
a backup of the jpg file archives elsewhere!*

## Troubleshooting Tips

> The first step of troubleshooting: identify the code and line number at which the error occurred
and inspect the code at that point. If the problem is addressed below, you'll need to understand 
the error source before you can identify the solution! If you have discovered a new error, you will 
have to investigate further and this is the best place to start.

### Common/Simple Problems

1.	The most common problems involve misnamed files or incorrect parameter choices. If 
the jobs fail immediately (within a few minutes of starting), that probably indicates 
an incorrect filename or path. Look carefully at the printed filepaths in the .out 
and .err files. These problems are often typos, so look closely!
2.	Another filename issue can occur after videos are created during the processing 
of analog signals. Apparatus and enclosure video folders and files should be named differently 
(e.g. moths and moths_free) during acquisition. If this is not the case, you will have to write 
a python or bash script to rename the files.


> Note: sometimes the final error will occur when creating the NWB file. However, this almost
always means some information was lost in the preceding steps (video creation or analog signal
processing). Look further up in the .err file to identify any errors from previous steps.

### Additional Errors
1.	If a CPU bottleneck occurred during data acquisition, you may find that there are mismatched
event counts and start times for the individual video events. If this is persistent throughout
the entire session, the data may not be salvageable. Intermittent bottlenecks are fixed prior to 
jpg2avi.py, but heavily corrupted event tracking is not correctible. This can indicate you need
to reduce the camera frame rates during acquisition, or perhaps the data SSDs on the acquisition computers
need to be reformatted to factory defaults (completely wiped), or in the worst case replaced.
2.	If the jumbled event counts happen near the end of either the apparatus or enclosure recordings,
you may find it easiest to delete events after the corruption began.
	-	For example, enclosure recordings should contain only 1 video event in a session. 
    A recent recording had very short over 1000 very short events at the end of the session. I
    deleted the corrupted events at the end using the following command:
    
			find /path/to/jpgs/ -type f -not -name *event_001* -delete

3. If the solid state hard drives become fragmented, you will likely encounter sessions containing a large number (potentially >1000) of video events that are very short and don't match with analog signals. Correcting these events is difficult but not impossible. Follow the example at for JL20231123, illustrated by [fix_episode_numbers_foraging_20231123_session1.py](/subject_specific_scripts/JL/analog_signal_processing_day_specific_files/fix_episode_numbers_foraging_20231123_session1.py) along with the manual annotation shown by [annotation_of_corrected_camera_events_JL_20231123.jpg](/subject_specific_scripts/JL/analog_signal_processing_day_specific_files/annotation_of_corrected_camera_events_JL_20231123.jpg). The manual annotation shows when large ranges of camera events need to be merged into the corrected event (e.g. correct_eIdx=10) and when a few events in a row can be corrected just by adjusting jpg filenames (e.g. correct_eIdx=11-16. This session was particularly fragmented, so you can also look at other records for simpler corrections (see [TEMPLATE_fix_episode_numbers_manually.py](/subject_specific_scripts/TEMPLATES/python/TEMPLATE_fix_episode_numbers_manually.py)) and there are further instructions as text within the python files. To get started, copy [check_timestamps_to_select_correct_event_for_event_corrections.py](/subject_specific_scripts/JL/analog_signal_processing_day_specific_files/check_timestamps_to_select_correct_event_for_event_corrections.py) into your subject_specific_files directory, edit the appropriate details, place an original set of jpgs in the correct directory in your scratch space and run the code in Spyder to produce a set of dataframes that will help with inspection. The best thing to look at is to match up camera events that have the exact same start time as the corrected event idx (see the times in the manual annotation record provided above). Once you have the corrections figured out, copy one `fix_episode_numbers_DETAILS.py` files into your new location, make the appropriate edits, and run it.  

>When this happens, you also need to de-fragment the hard drives. Do a factory reset on the drives. This will take multiple hours, and will wipe all the data, so make sure you have backed up all the data on the drive.
           
>Future users should add to this record as new problems are solved!






