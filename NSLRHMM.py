import logging
import numpy as np
from neuropype.engine import *

logger = logging.getLogger(__name__)


class NSLRHMM(Node):
    # --- Input/output ports ---
    data = Port(None, Packet, "Data to process.", required=True,
                editable=False, mutating=True)

    # --- Properties ---
    structural_error = FloatPort(0.1)
    optimize_noise = BoolPort(True)
    split_likelihood = StringPort()

    @classmethod
    def description(cls):
        return Description(name='NSLR-HMM',
                           description="""Naive Segmented Linear Regression - Hidden Markov Model.
                           A method for eye-movement signal denoising and segmentation,
                           and a related event classification method based on Hidden Markov Models.
                           
                           Need to install nslr and nslr-hmm.
                           First choice but fails: pip install git+https://github.com/pupil-labs/nslr.git
                           Instead use: pip install git+https://gitlab.com/nslr/nslr
                           then: pip install git+https://github.com/pupil-labs/nslr-hmm
                           """,
                           version='0.1',
                           license=Licenses.MIT)

    @data.setter
    def data(self, pkt):
        for n, chnk in enumerate_chunks(pkt, nonempty=True, only_signals=True, with_axes=(time,)):
            import nslr
            import nslr_hmm

            ts = chnk.block.axes[time].times
            # Segmentation using Pruned Exact Linear Time (PELT)
            segmentation = nslr.fit_gaze(ts, xs, structural_error=self.structural_error,
                                         optimize_noise=self.optimize_noise,
                                         split_likelihood=self.split_likelihood)
            seg_classes = nslr_hmm.classify_segments(segmentation.segments)
        self._data = pkt


"""
Notes.

Description:
https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-017-17983-x/MediaObjects/41598_2017_17983_MOESM1_ESM.pdf

Segmented linear regression implementations:
https://github.com/DataDog/piecewise/blob/master/piecewise/regressor.py#L12  (BSD)
https://gitlab.com/nslr/nslr/blob/master/nslr/slow_nslr.py (AGPL)

HMM gaze classification:
https://gitlab.com/nslr/nslr-hmm/blob/master/nslr_hmm.py

Usage example:
https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/eye_movement/worker/real_time_buffered_detector.py
"""