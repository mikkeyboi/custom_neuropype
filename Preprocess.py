"""Copyright (C) 2018 Intheon All rights reserved."""
# Metanode

import os
import logging
import pickle

from neuropype.engine import *
from neuropype.nodes.tensor_math import SelectRange
from neuropype.nodes.source_localization import AssignChannelLocations, RemoveUnlocalizedChannels
from neuropype.nodes.utilities import ExtractChannels, DejitterTimestamps, ShiftTimestamps
from neuropype.nodes.signal_processing import Rereferencing, FIRFilter, Resample
from neuropype.nodes.elementwise_math import Scaling
from neuropype.nodes.neural import BadChannelRemoval, ArtifactRemoval, InterpolateMissingChannels, RemoveBadTimeWindows
from neuropype.nodes.file_system import ExportH5
from neuropype.utilities import cache

logger = logging.getLogger(__name__)


class Preprocess(Node):
    """Meta node: Canonical Preprocessing chain"""

    # --- Input/output ports ---
    data = DataPort(Packet, "Data to process.")

    # skip this node in case of pre-processed data
    skip_preprocessing = BoolPort(False, """Skip all preprocessing, filtering or 
        cleaning of the data. Check this box if the data is already preprocessed 
        and ready for analysis. This node will still however select the channel 
        subset specified here (if any).""")

    # --- Properties ---
    channels_to_import = Port("all", object, """List of channels to retain. Can be a list of channel
        names or indices, or a string. If the latter, regular Python list and range syntax is 
        supported, as well as comma-separated or space-separated lists of channel names or indices 
        lists. The special keyword "all" stands for all channels.""")
    scale_factor = FloatPort(1.0, None, """Scaling factor. The signal is scaled by this 
        factor. Can be used for manual unit conversion.""", expert=True)

    lowpass_frequencies = ListPort([45, 50], float, """High noise frequencies 
        to be filtered out with a lowpass filter. You can either give the
        cutoff frequency as a single value, or two frequencies (separated 
        by a comma), to specify the rolloff curve.""")
    lowpass_stop_atten = FloatPort(120.0, None, """Minimum attenuation in stopband 
        for the lowpass filter. This is the minimum acceptable attenuation, 
        in dB, in the stopband, which is ideally infinitely suppressed, 
        but in practice 30-80 dB are often enough unless there is enormous drift (e.g., with some 
        dry headsets). """, expert=True)

    highpass_frequencies = ListPort([0.1, 0.5], float, """Low noise frequencies 
        to be filtered out with a highpass filter. You can either give the
        cutoff frequency as a single value, or two frequencies (separated 
        by a comma), to specify the rolloff curve.""")
    highpass_stop_atten = FloatPort(50.0, None, """Minimum attenuation in stopband 
        for the highpass filter. This
        is the minimum acceptable attenuation, in dB, in the stopband,
        which is ideally infinitely suppressed, but in practice 30-80 dB are
        enough, depending on the amplitudes of the signals to attenuate.
        """, expert=True)

    rereferencing = Port('CAR', object, """Rereferencing. For common average referencing, 
        enter: CAR. To reference to certain channels, a list of channel labels can be given (as a
        Python list, or in an ad-hoc comma-separated or space-separated list of channel labels).""")

    sampling_rate = FloatPort(None, None, """Target sampling rate (in Hz).""",
                              verbose_name='desired sampling rate')

    bad_channel_corr_threshold = FloatPort(0.8, [0, 0.3, 0.95, 1], """Correlation threshold.
        Higher values (above 0.7) are more stringent and will remove more
        channels (i.e., moderately bad channels get removed). Values below
        0.6 would be considered very lax (i.e., only the worst channels get
        removed). This threshold is based on the correlation between a
        channel and what one would expect the channel to be given the other
        channels. Note that this parameter is only used when channel
        locations are available.""", verbose_name='correlation threshold')
    bad_channel_noise_threshold = FloatPort(4, None, """High-frequency noise threshold.
        Lower values (below 3.5) are more stringent and will remove more
        channels (i.e., moderately bad channels will get removed). Values
        above 5 would be considered very lax (i.e., only the worst channels
        get removed). This threshold is based on the amount of high frequency
        noise compared to other channels, and is measured in standard
        deviations.""")

    burst_removal_cutoff = FloatPort(10.0, [0, 2.5, 20.0, 30.0], """Threshold for burst-type 
        artifact removal, in standard deviations. Data portions whose amplitude is
        larger than this threshold (relative to the calibration data) are
        considered bad data and will be removed. The most aggressive
        value that can be used without losing too much EEG is 3. A quite
        conservative value would be 10.0.""")

    # shared by bad channel and artifact removal
    use_clean_window = BoolPort(False, """Use clean time windows to calibrate artifact
        removal thresholds. This applies to all artifact removal steps that involve such a 
        calibration, including bad-channel removal and burst-artifact removal.
        """, expert=True)
    calib_seconds = IntPort(0, None, """Minimum amount of data to gather for
        calibration. When this filter is run online and has not yet been
        calibrated, then it will first buffer this many seconds of data in order
        to compute its measures before any output is produced. (This is used 
        both for bad channel removal and artifact removal.) 
        """)
    init_on = ListPort([], object, """Time range to initialize on.
        If two numbers are given, either in seconds, or as fractions of the
        calibration data (if both below 1), then the filter will be initialized
        only on that subset of the calibration data. Another use case is to
        select an initial baseline period in a longer recording, in order to
        avoid having to re-run the (fairly expensive) filter in each fold of a
        cross-validation.
        """, expert=True, verbose_name='initialize on this time range')

    # bad window  removal
    clean_signal_range = ListPort([-4, 6], object, """Minimum and maximum of
        clean signal range, in multiples of standard deviation. The
        minimum and maximum standard deviations within which the power of a
        channel must lie (relative to a robust estimate of the clean EEG
        power distribution in the channel) for it to be considered not bad.
        Use values between [-10 15].""")
    max_bad_channels = FloatPort(0.20, None, """Maximum fraction of bad
        channels allowed . The maximum fraction of bad channels that a
        retained window may still contain (more than this and it is removed).
        Reasonable range is 0.05 (very clean output) to 0.3 (very lax
        cleaning of only coarse artifacts).
        """, expert=True)

    use_caching = BoolPort(False, """Enable caching. This will significantly 
        speed up re-use of the same data.""")
    export_to_h5 = BoolPort(False, """Export processed data to HDF5 format. 
        (Files will be named and saved in the same folder as the original files. """)

    def __init__(self, **kwargs):
        """Create a new node. Accepts initial values for the ports."""
        super().__init__(**kwargs)

    @classmethod
    def description(cls):
        """Declare descriptive information about the node."""
        return Description(name='Preprocess (Meta)',
                           description="""\
                           A meta-node that performs a standard preprocessing
                           chain including high/low pass filtering, artifact 
                           and bad channel removal, referencing, 
                           etc.""",
                           version='1.0.0', status=DevStatus.beta)

    # noinspection PyCallingNonCallable
    @data.setter
    def data(self, v):
        data = v

        # select desired subset of channels
        channels_to_import = ':' if self.channels_to_import in ['(all)', 'all'] else self.channels_to_import
        data = SelectRange(axis='space', selection=channels_to_import, unit='auto')(data)

        if not self.skip_preprocessing:
            record = cache.try_lookup(context=self, enabled=self.use_caching,
                                      verbose=True, data=v)
            if record.success():
                data = record.data
            else:
                # misc pre-processing steps
                data = DejitterTimestamps()(data)
                data = Scaling(factor=self.scale_factor)(data)

                # handle channel locations, also remove channels that don't have locations
                data = AssignChannelLocations()(data)
                data = RemoveUnlocalizedChannels()(data)
                orig_channels = ExtractChannels()(data)
                print(orig_channels)

                # zero-mean
                data = Rereferencing(axis='time', reference_unit='indices', estimator='median')(data)

                # resampling (earlier as it can speed up pre-processing)
                data = Resample(rate=self.sampling_rate)(data)

                # highpass filter
                data = FIRFilter(frequencies=self.highpass_frequencies, mode='highpass', minimum_phase=False,
                                 stop_atten=self.highpass_stop_atten)(data)

                # artifact removal
                data = BadChannelRemoval(calib_seconds=self.calib_seconds, corr_threshold=self.bad_channel_corr_threshold,
                                         noise_threshold=self.bad_channel_noise_threshold,
                                         use_clean_window=self.use_clean_window)(data)
                data = ArtifactRemoval(cutoff=self.burst_removal_cutoff, calib_seconds=self.calib_seconds,
                                       use_clean_window=self.use_clean_window)(data)

                # uses the channel names from ExtractChannels()
                data = InterpolateMissingChannels(desired_channels=orig_channels)(data)

                # re-referencing
                reref_range = ':' if self.rereferencing.upper() == 'CAR' else self.rereferencing
                data = Rereferencing(reference_range=reref_range, reference_unit='auto')(data)

                data = RemoveBadTimeWindows(zscore_thresholds=self.clean_signal_range,
                                            max_bad_channels=self.max_bad_channels)(data)

                # lowpass and remove data points above cutoff
                data = FIRFilter(frequencies=self.lowpass_frequencies,
                                 mode='lowpass', minimum_phase=False,
                                 stop_atten=self.lowpass_stop_atten)(data)

                # fix up filter lags
                data = ShiftTimestamps(compensate_filter_lag=True)(data)

                record.writeback(data=data)

            if self.export_to_h5:
                h5filename = data.chunks['eeg'].props['source_url'].replace('file://','') + '.h5'
                ExportH5(filename=h5filename)(data)

        self._data = data
