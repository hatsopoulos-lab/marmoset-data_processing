# This should be filled out by a current member of the lab with details of how to do this on the devoted spike-sorting machine.

After the steps to be filled in above, you will be ready to add spike-sorted data manually curated with Phy2 to a `DATASET_DETAILS_processed.nwb` file.

## Add spike-sorted data to processed NWB file

Use [TEMPLATE_add_phy_sorting_to_nwb.py](/subject_specific_scripts/TEMPLATES/python/TEMPLATE_add_phy_sorting_to_nwb.py) to add processed pose data to a new or existing `DATASET_DETAILS_processed.nwb` file. There are instructions for editing the necessary info and filepaths at the top of the file.

>Note: there is a "removed_chans" variable that likely only applies to old datasets. Some of the earliest datasets had channels removed because they were dead/outlier channels. I believe this step is deprecated, so all new data should have an empty list for this variable.

## Validate processed NWB file

Use [TEMPLATE_validate_nwb_files.py](/subject_specific_scripts/TEMPLATES/python/TEMPLATE_validate_nwb_files.py)
This will plot example spiketimes for each unit indicated on the raw signal, and can be used to identify timing errors or errors in unit-signal alignment. Timing errors will be seen as spiketimes that don't align closely with the obvious spikes in the raw signal (but in a systematic way that hsould happen for every unit). Misalignment between unit and signal will be seen as example plots that seem to have no correlation b/w signal and spiketime after a particular unit number. This may indicate that the `removed_chans` list discussed in the note above is incorrect and needs to be fixed. Use the channel numbers in the plot titles to figure out how to fix this. If multiple channels were removed, you'll have to go one-by-one, inspecting the spike-sort until all channels/units are aligned.