#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:32:48 2022

@author: daltonm
"""

import argparse
from datetime import datetime
from dateutil import tz
from pathlib import Path
from neuroconv.datainterfaces import BlackrockRecordingInterface

def create_nwb_and_store_raw_neural_data(file_path):

    nwbfile_path = file_path.replace('.ns6', '.nwb')

    if Path(nwbfile_path).is_file():
        print('\nNWB file already exists at %s' % nwbfile_path)    
        return

    # For this interface we need to pass the location of the ``.ns6`` file
    interface = BlackrockRecordingInterface(file_path=file_path, verbose=True)
    
    # Extract what metadata we can from the source files
    metadata = interface.get_metadata()
    # For data provenance we add the time zone information to the conversion
    session_start_time = datetime.fromisoformat(metadata["NWBFile"]["session_start_time"])
    session_start_time = session_start_time.replace(tzinfo=tz.gettz("US/Central")).isoformat()
    metadata["NWBFile"].update(session_start_time=session_start_time)
    
    # Run the conversion
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)
    
    # If the conversion was successful this should evaluate to ``True`` as the file was created.
    if Path(nwbfile_path).is_file():
        print('\nNWB file successfully created with raw neural signals at %s' % nwbfile_path)
    else:
        print('\nAn error occurred while trying to create NWB file at %s' % nwbfile_path)

    return

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file_path", required=True, type=str,
        help="path to ns6 file that will be stored in new NWB file, e.g. /project/nicho/data/marmosets/electrophys_data_for_processing/TY20221024_testbattery/TY20221024_testbattery_001.ns6")
    args = vars(ap.parse_args())
    
    # args = {'file_path' : '/project/nicho/data/marmosets/electrophys_data_for_processing/TY20221024_testbattery/TY20221024_testbattery_001.ns6'}
    
    create_nwb_and_store_raw_neural_data(args['file_path'])