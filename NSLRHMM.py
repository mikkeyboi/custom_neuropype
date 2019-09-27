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

    @classmethod
    def description(cls):
        return Description(name='NSLR-HMM',
                           description="""Naive Segmented Linear Regression - Hidden Markov Model.
                           A method for eye-movement signal denoising and segmentation,
                           and a related event classification method based on Hidden Markov Models.
                           
                           Need to install nslr and nslr-hmm.
                           First choice (Win/Linux): pip install git+https://github.com/pupil-labs/nslr.git
                           Mac: pip install git+https://gitlab.com/nslr/nslr
                           then: pip install git+https://github.com/pupil-labs/nslr-hmm
                           """,
                           version='0.1',
                           license=Licenses.MIT)

    @data.setter
    def data(self, pkt):
        for n, chnk in enumerate_chunks(pkt, nonempty=True, only_signals=True, with_axes=(time,)):
            import nslr
            import nslr_hmm
            import pandas as pd

            ts = chnk.block.axes[time].times
            xs = chnk.block[time, ...].data
            # Segmentation using Pruned Exact Linear Time (PELT)
            segmentation = nslr.fit_gaze(ts, xs, structural_error=self.structural_error,
                                         optimize_noise=self.optimize_noise)
            seg_classes = nslr_hmm.classify_segments(segmentation.segments)

            if False:
                COLORS = {
                    nslr_hmm.FIXATION: 'blue',
                    nslr_hmm.SACCADE: 'black',
                    nslr_hmm.SMOOTH_PURSUIT: 'green',
                    nslr_hmm.PSO: 'yellow',
                }
                import matplotlib.pyplot as plt
                plt.plot(ts, xs[:, 0], '.')
                for seg_ix, seg in enumerate(segmentation.segments):
                    plt.plot(seg.t, np.array(seg.x)[:, 0],
                             linestyle='--', linewidth=2,
                             color=COLORS[seg_classes[seg_ix]])
                plt.show()

            seg_class_str_map = {
                nslr_hmm.FIXATION: 'Fixation',
                nslr_hmm.SACCADE: 'Saccade',
                nslr_hmm.SMOOTH_PURSUIT: 'SmoothPursuit',
                nslr_hmm.PSO: 'PSO'
            }

            seg_ts = np.stack([(_.t[0], _.t[-1]) for _ in segmentation.segments], axis=-1)
            seg_xs = np.stack([np.vstack((_.x[0], _.x[-1])) for _ in segmentation.segments], axis=-1)
            ev_dict = {
                'StartTime': seg_ts[0],
                'EndTime': seg_ts[1],
                'Marker': [seg_class_str_map[_] for _ in seg_classes],
                'Duration': seg_ts[1] - seg_ts[0],
                'Amp': np.linalg.norm(seg_xs[1, :, :] - seg_xs[0, :, :], axis=0),
            }
            for dim_ix, dim_name in enumerate(['X', 'Y']):
                ev_dict.update({
                    'Start' + dim_name: seg_xs[0, dim_ix, :],
                    'Pos' + dim_name: seg_xs[1, dim_ix, :]
                })

            ev_df = pd.DataFrame(ev_dict)

            ev_dat = ev_df.drop(['StartTime'], axis=1).to_records(index=False)
            ev_blk = Block(data=np.nan * np.ones((len(ev_dat),)),
                           axes=(InstanceAxis(ev_df['StartTime'], data=ev_dat),))

            pkt.chunks[n] = Chunk(block=ev_blk, props=[Flags.is_event_stream])

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