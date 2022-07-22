import numpy as np
from phy import IPlugin

# https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html


class qualityMetrics(IPlugin):
    def attach_to_controller(self, controller):

        def isi_violations(cluster_id):
            # Modified from Spike interface spike metrics:
            # https://github.com/SpikeInterface/spikemetrics/
            # spikeinterface/spikeinterface/toolkit/qualitymetrics/misc_metrics.py

            min_isi = 0
            isi_threshold = 1.5/1000

            spike_train = controller.get_spike_times(cluster_id).data
            isis = np.diff(spike_train)
            duration = spike_train[-1]-spike_train[0]
            num_violations = sum(isis < isi_threshold)
            num_spikes = len(spike_train)
            violation_time = 2 * num_spikes * (isi_threshold - min_isi)
            total_rate = num_spikes/duration
            violation_rate = num_violations / violation_time
            fpRate = violation_rate / total_rate

            return fpRate

        def snr(cluster_id):
            # Modified from https://github.com/SpikeInterface/spikeinterface/toolkit/utils.py (noise level)
            # spikeinterface/spikeinterface/toolkit/qualitymetrics/misc_metrics.py
            ch = controller.get_best_channel(cluster_id)

            # get noise level
            amps = controller.get_amplitudes(cluster_id)
            # amps = controller._get_waveforms(cluster_id).data
            med = np.median(amps, axis=0, keepdims=True)
            noise_levels = np.median(
                np.abs(amps - med), axis=0) / 0.6745  # MAD Estimation

            snr = np.mean(np.abs(amps))/noise_levels

            return snr

        # Use this dictionary to define custom cluster metrics.
        # We memcache the function so that cluster metrics are only computed once and saved
        # within the session, and also between sessions (the memcached values are also saved
        # on disk).
        controller.cluster_metrics['isi_violations'] = controller.context.memcache(
            isi_violations)

        controller.cluster_metrics['snr'] = controller.context.memcache(snr)
