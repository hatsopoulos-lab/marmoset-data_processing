#!/usr/bin/env python
# coding: utf-8

# # How to extract data from NWB
# 
# This notebook will demonstrate opening nwb files, exploring them, and extracting data that is important to our lab.

# ## But first...consider checking something off the outstanding to-do list!
# 
#     1) Write script/functions to add LFP to "processing['ecephys']" module
#     2) Convert all old TY neural + cams data to nwb format
#     3) Update neural_dropout_first_pass.py to use raw .ns6 data rather than .nev data. Important especially for sleep sessions
#     4) Add SNR, mean_waveform, and ISI data to units table. This may include adding some/all of these to phy by figuring out how to include the plugins written by Marina
#     5) Add touchscreen timestamps to processing scripts and add them to NWB intervals

# ### Import modules

# In[33]:


from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget
import ndx_pose
import numpy as np
import matplotlib.pyplot as plt


# ### Define nwbfile path and open it in read mode

# In[34]:


#nwb_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003.nwb'
#nwb_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/MG20230416_1505_mothsAndFree/MG20230416_1505_mothsAndFree-002_acquisition.nwb'
nwb_acquisition_file = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003_acquisition.nwb'
nwb_processed_file   = '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20210211_freeAndMoths/TY20210211_freeAndMoths-003_processed.nwb' 

io_acq = NWBHDF5IO(nwb_acquisition_file, mode='r')
nwb_acq = io_acq.read()

io_prc = NWBHDF5IO(nwb_processed_file, mode='r')
nwb_prc = io_prc.read()


# ### Use nwb2widget to explore the data

# In[35]:


nwb2widget(nwb_acq)


# ### Look at Notes, other metadata

# In[36]:


print(nwb_prc.notes)
print('\n\n\n\n')
print(nwb_acq.acquisition)


# ### Get some neural data and perform some checks

# In[37]:


elabel = 'elec62'

# create timestamps for raw neural data from starting_time, rate, and data shape
start = nwb_acq.acquisition['ElectricalSeriesRaw'].starting_time
step = 1/nwb_acq.acquisition['ElectricalSeriesRaw'].rate
stop = start + step*nwb_acq.acquisition['ElectricalSeriesRaw'].data.shape[0]
raw_timestamps = np.arange(start, stop, step)

# get sorted units information, extract spike_times
units = nwb_prc.units.to_dataframe()
unit_to_plot = units.loc[units.electrode_label == elabel, :]
spike_times = unit_to_plot.spike_times.iloc[0]

# Get electrodes table, extract the channel index matching the desired electrode_label
raw_elec_table = nwb_acq.acquisition['ElectricalSeriesRaw'].electrodes.to_dataframe()
raw_elec_index = raw_elec_table.index[raw_elec_table.electrode_label == elabel]

# Get first 100000 samples raw data for that channel index
raw_data_single_chan = nwb_acq.acquisition['ElectricalSeriesRaw'].data[:100000, raw_elec_index.values]


# ##### Pull out data around spike time in raw neural data (using tMod = 0 or tMod = nwbfile.acqusition['ElectricalSeriesRaw'] starting time)

# In[38]:


tMod = 0 #nwb_acq.acquisition['ElectricalSeriesRaw'].starting_time
spikes_indexed_in_raw = [np.where(np.isclose(raw_timestamps, spk_time+tMod, atol=1e-6))[0][0] for spk_time in spike_times[:10]]


# In[39]:


spkNum = 1
plt.plot(raw_timestamps[spikes_indexed_in_raw[spkNum] - 100 : spikes_indexed_in_raw[spkNum] + 100], 
         raw_data_single_chan[spikes_indexed_in_raw[spkNum] - 100 : spikes_indexed_in_raw[spkNum] + 100])
plt.plot(raw_timestamps[spikes_indexed_in_raw[spkNum]], raw_data_single_chan[spikes_indexed_in_raw[spkNum]], 'or')


# ### Look at an individual reaching segment and link it to the correct kinematics

# In[40]:


segment_idx = 39

# get info in dataframe for specific segment_idx
segment_df = nwb_prc.intervals['reaching_segments_moths'].to_dataframe()
segment_info = segment_df.iloc[segment_idx]

# get event data using container and ndx_pose names from segment_info table following form below:
# nwb.processing['goal_directed_kinematics'].data_interfaces['moths_s_1_e_004_position']
event_data = nwb_prc.processing[segment_info.kinematics_module].data_interfaces[segment_info.video_event] 
hand_kinematics = event_data.pose_estimation_series['hand'].data[:] 
timestamps      = event_data.pose_estimation_series['hand'].timestamps[:]
reproj_error    = event_data.pose_estimation_series['hand'].confidence[:]

# plot full_event 
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(timestamps, hand_kinematics)
axs[0].vlines(x=[segment_info.start_time, segment_info.stop_time], ymin=-3,ymax=14, colors='black', linestyle='dashdot')
axs[1].plot(timestamps, reproj_error, '.b')
axs[0].set_ylabel('Position (cm) for x (blue), y (orange), z (green)')
axs[0].set_title('Entire video event hand kinematics')
axs[1].set_ylabel('Reprojection Error b/w Cameras (pixels)')
axs[1].set_xlabel('Time (sec)')
plt.show()

# extract kinematics of this single reaching segment and plot
reach_hand_kinematics = hand_kinematics[segment_info.start_idx:segment_info.stop_idx]
reach_reproj_error    = reproj_error   [segment_info.start_idx:segment_info.stop_idx]
reach_timestamps      = timestamps     [segment_info.start_idx:segment_info.stop_idx]
peak_idxs = segment_info.peak_extension_idxs.split(',')
peak_idxs = [int(idx) for idx in peak_idxs]
peak_timestamps = timestamps[peak_idxs]
peak_ypos = hand_kinematics[peak_idxs, 1]

# plot single reaching segment 
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(reach_timestamps, reach_hand_kinematics)
axs[0].plot(peak_timestamps, peak_ypos, 'or')
axs[1].plot(reach_timestamps, reach_reproj_error, '.b')
axs[0].set_ylabel('Position (cm) for x (blue), y (orange), z (green)')
axs[0].set_title('Reaching segment hand kinematics')
axs[1].set_ylabel('Reprojection Error b/w Cameras (pixels)')
axs[1].set_xlabel('Time (sec)')
plt.show()


# In[41]:


# get table of sorted unit info
units_df = nwb_prc.units.to_dataframe()
elec_positions = units_df.loc[:, ['x', 'y', 'z', 'electrode_label']]
elec_positions


# ### Load and isolate analog channels using electrodes table

# In[42]:


raw = nwb_acq.acquisition['ElectricalSeriesRaw']

start = raw.starting_time
step = 1/raw.rate
stop = start + step*raw.data.shape[0]
raw_timestamps = np.arange(start, stop, step)

elec_df = raw.electrodes.to_dataframe()
analog_idx = [idx for idx, name in elec_df['electrode_label'].iteritems() if 'ainp' in name]
electrode_labels = elec_df.loc[analog_idx, 'electrode_label']

# plot the first 3 minutes of data for the channels
time_to_plot = 3*60
num_samples = int(raw.rate * time_to_plot)
num_channels = np.min([2, len(analog_idx)])
fig, axs = plt.subplots(num_channels, 1, sharex=True) 
for cIdx in range(num_channels):
    analog_signals = raw.data[:num_samples, analog_idx[cIdx]] * elec_df['gain_to_uV'][analog_idx[cIdx]] * raw.conversion
    axs[cIdx].plot(raw_timestamps[:num_samples], analog_signals)
    axs[cIdx].set_title(electrode_labels.iloc[cIdx])
    axs[cIdx].set_ylabel('Raw Signal (V)')

axs[cIdx].set_xlabel('Timestamps (sec)')
    
plt.show()


# ### Now for a few neural channels

# In[43]:


raw = nwb_acq.acquisition['ElectricalSeriesRaw']
elec_df = raw.electrodes.to_dataframe()
analog_idx = [idx for idx, name in elec_df['electrode_label'].iteritems() if 'elec' in name]
electrode_labels = elec_df.loc[analog_idx, 'electrode_label']

# plot the first 3 minutes of data for the channels
time_to_plot = 3*60
num_samples = int(raw.rate * time_to_plot)
num_channels = np.min([3, len(analog_idx)])
fig, axs = plt.subplots(num_channels, 1, sharex=True) 
for cIdx in range(num_channels):
    analog_signals = raw.data[:num_samples, analog_idx[cIdx]] * elec_df['gain_to_uV'][analog_idx[cIdx]] * raw.conversion
    axs[cIdx].plot(raw_timestamps[:num_samples], analog_signals)
    axs[cIdx].set_title(electrode_labels.iloc[cIdx])

axs[cIdx].set_ylabel('Raw Signal (V)')
axs[cIdx].set_xlabel('Timestamps (sec)')
    
plt.show()


# ### When you finish working with the data, close the files

# In[44]:


io_acq.close()
io_prc.close()

