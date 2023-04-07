#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:34:05 2023

@author: daltonm
"""

# import toolboxes
import numpy as np
import re
import pynwb
import datetime
import pytz
import matplotlib.pyplot as plt
from pathlib import Path

from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe, plot_probe_group

def read_prb_hatlab(file):
    """
    Read a PRB file and return a ProbeGroup object.
    Since PRB does not handle contact shapes, contacts are set to be circle of 5um radius.
    Same for the probe shape, where an auto shape is created.
    PRB format does not contain any information about the channel of the probe
    Only the channel index on device is given.
    Parameters
    ----------
    file : Path or str
        The file path
    Returns
    --------
    probegroup : ProbeGroup object
    """

    file = Path(file).absolute()
    assert file.is_file()
    with file.open("r") as f:
        contents = f.read()
    contents = re.sub(r"range\(([\d,]*)\)", r"list(range(\1))", contents)
    prb = {}
    exec(contents, None, prb)
    prb = {k.lower(): v for (k, v) in prb.items()}

    if "channel_groups" not in prb:
        raise ValueError("This file is not a standard PRB file")

    probegroup = ProbeGroup()
    imp = []
    for i, group in prb["channel_groups"].items():
        ndim = 2
        probe = Probe(ndim=ndim, si_units="um")

        try:
            chans = np.array(group["channels"], dtype="int64")
        except:
            chans = np.array(group["channels"], dtype=str)
            
        try:
            positions = np.array([group["geometry"][c] for c in chans], dtype="float64")
        except:
            positions = np.array([group["geometry"][idx] for idx, c in enumerate(chans)], dtype="float64")

        # try:
        #     chan_labels = np.array([group["chanels"][c] for c in chans], dtype="float64")
        # except:
        #     chan_labels = np.array([group["chan_label"][idx] for idx, c in enumerate(chans)], dtype="float64")        
        
        num_contacts = positions.shape[0]
        plane_axes = np.zeros((num_contacts, 2, ndim))
        plane_axes[:, 0, 0] = 1
        plane_axes[:, 1, 1] = 1
        
        probe.set_contacts(
            positions=positions, shapes="circle", shape_params={"radius": prb['radius']}, shank_ids=chans, plane_axes = plane_axes
        )
        # probe.create_auto_shape(probe_type="tip")

        probegroup.add_probe(probe)
        
        imp.append(np.array(group['impedance'][0], dtype=str))

    return probegroup, imp

def plot_prb(probegroup):
    probegroup_df = probegroup.to_dataframe()
    probenum = list(probegroup_df['shank_ids'])
    probenum = [str(prb).split('elec')[-1] for prb in probenum]
    plot_probe_group(probegroup, same_axes=True, with_channel_index=False)
    ax = plt.gca()
    for idx, prb in enumerate(probenum):
        try:
            ax.text(probegroup_df['x'][idx], probegroup_df['y'][idx], probegroup_df['z'][idx], prb)
        except:
            ax.text(probegroup_df['x'][idx], probegroup_df['y'][idx], prb)

    plt.show()
