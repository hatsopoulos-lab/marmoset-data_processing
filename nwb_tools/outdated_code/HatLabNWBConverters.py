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

    
