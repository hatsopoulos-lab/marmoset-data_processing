"""
HatLab NWB Converters: This toolbox should contain an NWB Converters for each
experiment type used in the lab. Each converter specifies the datainterfaces
used for each of the types of raw data acquired in the experiment. The idea is
that the raw data and metadata about the data collection will be combined and
stored in an NWB file immediately after data collection. Future processes on
the raw data will be subsequently appended to the NWB file, creating a record
of the data used to in publications and analysis.

v0.0a PLA 122321
"""

# import needed toolboxes
from nwb_conversion_tools import NWBConverter
from nwb_conversion_tools import BlackrockRecordingExtractorInterface
from nwb_conversion_tools import BlackrockSortingExtractorInterface
from nwb_conversion_tools import MovieInterface
from pynwb import NWBHDF5IO
from ndx_pose import PoseEstimationSeries, PoseEstimation
import re
import numpy as np

# Marmoset check recordings converter
class MarmCheckNWBConverter(NWBConverter):
    """
    A converter for the 5min morning and evening recording checks. This
    experiment type has no videos or behavioral data associated with the
    acquisition. It assumes an NS6 file and possibly a NEV from online
    sorting.

    """

    data_interface_classes = dict(
        BlackrockRecordingInterfaceRaw=BlackrockRecordingExtractorInterface,
        BlackrockSortingInterface=BlackrockSortingExtractorInterface
    )

    def get_conversion_options(self):
        conversion_options = dict()
        if 'BlackrockRecordingInterfaceRaw' in self.data_interface_objects:
            conversion_options['BlackrockRecordingInterfaceRaw'] = dict(
                write_as='raw',
                es_key='ElectricalSeries_raw',
                stub_test=False
            )

        if 'BlackrockSortingInterface' in self.data_interface_objects:
            conversion_options['BlackrockSortingInterface'] = dict()
        return conversion_options

class MarmForageNWBConverter(NWBConverter):
    """
    A converter for Marmoset foraging session. The raw data associated
    with this experiment should be NS6 file, NEV file and videos from
    multiple cameras.
    """

    data_interface_classes = dict(
        BlackrockRecordingInterfaceRaw=BlackrockRecordingExtractorInterface,
        BlackrockSortingInterface=BlackrockSortingExtractorInterface,
        KinematicVideos=MovieInterface,
        CalibrationVideos=MovieInterface,
    )

def MarmAnipose2NWB(nwbfile_path, trajectory_data, signal_data): 
    
    marker_names = trajectory_data['marker_names']
    experiment = trajectory_data['experiment']
    
    if experiment == 'free':
        description = 'free, unconstrained behavior in the home enclosure'
    elif experiment == 'foraging':
        description = 'foraging kinematics in the apparatus'
    elif experiment == 'crickets' or experiment == 'moths':
        description = 'prey capture kinematics - %s' % experiment
    elif experiment.lower() == 'betl':
        description = 'kinematics of virtual prey capture in BeTL task'
    else:
        description = 'kinematics for %s experiment' % experiment
        
    with NWBHDF5IO(nwbfile_path, 'r+') as io:
        nwbfile = io.read()
        
        behavior_pm = nwbfile.create_processing_module(
            name='behavior_kinematics',
            description=description
        )
        
        sessPattern = re.compile('[0-9]{3}.nwb') 
        sessNum = int(re.findall(sessPattern, nwbfile_path)[-1][:3])
        for frame_times, event_info in zip(signal_data['frameTimes_byEvent'], signal_data['event_info']):
            for eventIdx, timestamps in enumerate(frame_times):
                if event_info.ephys_session[eventIdx] == sessNum:
                    series_name = '%s_s_%s_e_%s' % (event_info.exp_name[eventIdx], 
                                                    event_info.video_session[eventIdx], 
                                                    str(eventIdx + 1).zfill(3)) 
                    
                    # start_time = event_info.start_time[eventIdx]
                    
                    # data = np.ones((len(timestamps), 3, 31))
                    
                    # position_series.create_timeseries(name = series_name,
                    #                                   data = data,
                    #                                   unit = 'm',
                    #                                   conversion = 1e-2,
                    #                                   timestamps = timestamps,
                    #                                   description = 'Dimensions of data are [time, x/y/z, marker]. The markers are...',
                    #                                   continuity = 'continuous')
                    
                    pose_estimation_series = []                    
                    for mIdx, mName in enumerate(marker_names):
                        data = np.ones((len(timestamps), 3))
                        confidence = np.random.rand(len(timestamps))  # a confidence value for every frame
                        marker_pose = PoseEstimationSeries(
                            name=mName,
                            description='Marker placed at ___',
                            data=data,
                            unit='m',
                            conversion = 1e-2,
                            reference_frame='(0,0,0) corresponds to the near left corner of the prey capture/foraging arena or touchscreen, viewed from the marmoset perspective',
                            timestamps=timestamps,  # link to timestamps of front_left_paw
                            confidence=confidence,
                            confidence_definition='Reprojection error output from Anipose',
                        )
            
                        pose_estimation_series.append(marker_pose) 
        
                    pe = PoseEstimation(
                        pose_estimation_series=pose_estimation_series,
                        name = series_name,
                        description='Estimated positions of all markers using DLC+Anipose, with post-Anipose cleanup',
                        original_videos=['camera1.mp4', 'camera2.mp4'],
                        labeled_videos=['camera1_labeled.mp4', 'camera2_labeled.mp4'],
                        dimensions=np.array([[1440, 1080], [1440, 1080]], dtype='uint8'),
                        scorer='DLC_resnet50_openfieldOct30shuffle1_1600',
                        source_software='DeepLabCut+Anipose',
                        source_software_version='2.2b8',
                        nodes=marker_names,
                        edges=np.array([[0, 1]], dtype='uint8'),
                        # devices=[camera1, camera2],  # this is not yet supported
                    )
                    
                    behavior_pm.add(pe)

        # read_nwbfile.add_acquisition(test_ts)
    
        # write the modified NWB file
        # behavior_pm.add(position_series)
        io.write(nwbfile)
    
