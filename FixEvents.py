import logging
import scipy.io
import numpy as np
from neuropype.engine.packet import Chunk
from neuropype.engine.block import Block
from neuropype.engine.axes import InstanceAxis
from neuropype.engine.constants import Licenses, Flags
from neuropype.engine.packet import Packet
from neuropype.engine.node import Node, Description
from neuropype.engine.ports import Port, StringPort, EnumPort
from neuropype.utilities.cloud import storage


logger = logging.getLogger(__name__)


class FixEvents(Node):
    # --- Input/output ports ---
    data = Port(None, Packet, "Data to process.", required=True,
                editable=False, mutating=True)
    filename = StringPort("", """Name of the event dataset.
                    """, is_filename=True)

    # options for cloud-hosted files
    cloud_host = EnumPort("Default", ["Default", "Azure", "S3", "Google",
                                      "Local", "None"], """Cloud storage host to
            use (if any). You can override this option to select from what kind of
            cloud storage service data should be downloaded. On some environments
            (e.g., on NeuroScale), the value Default will be map to the default
            storage provider on that environment.""")
    cloud_account = StringPort("", """Cloud account name on storage provider
            (use default if omitted). You can override this to choose a non-default
            account name for some storage provider (e.g., Azure or S3.). On some
            environments (e.g., on NeuroScale), this value will be
            default-initialized to your account.""")
    cloud_bucket = StringPort("", """Cloud bucket to read from (use default if
            omitted). This is the bucket or container on the cloud storage provider
            that the file would be read from. On some environments (e.g., on
            NeuroScale), this value will be default-initialized to a bucket
            that has been created for you.""")
    cloud_credentials = StringPort("", """Secure credential to access cloud data
            (use default if omitted). These are the security credentials (e.g.,
            password or access token) for the the cloud storage provider. On some
            environments (e.g., on NeuroScale), this value will be
            default-initialized to the right credentials for you.""")

    @classmethod
    def description(cls):
        return Description(name='Fix Events for Location Rule',
                           description="""
                           """,
                           version='0.1',
                           license=Licenses.MIT)

    @data.setter
    def data(self, packet):
        if packet is not None:
            filename = storage.cloud_get(self.filename, host=self.cloud_host,
                                         account=self.cloud_account,
                                         bucket=self.cloud_bucket,
                                         credentials=self.cloud_credentials)

            logger.info("Replacing markers with events loaded from %s..." % filename)
            mat = scipy.io.loadmat(filename)

            ev_times = []
            ev_strs = []
            for tr_ix in range(mat['startTime'].size):
                start = mat['startTime'][0][tr_ix]
                target_onset = mat['targetOnset'][0][tr_ix]
                cue_onset = mat['cueOnset'][0][tr_ix]
                sac_onset = mat['sacStartTime'][0][tr_ix]
                class_id = int(mat['newClass'][0][tr_ix])
                ev_times.extend([start + target_onset, start + cue_onset, start + sac_onset])
                ev_strs.extend(['Target-' + str(class_id), 'Cue-' + str(class_id), 'Saccade-' + str(class_id)])

            if len(ev_times) > 0:
                ev_times = np.asarray(ev_times) / 1000
                marker_block = Block(data=np.nan * np.ones_like(ev_times),
                                     axes=(InstanceAxis(ev_times,
                                                        data=np.asarray(ev_strs),
                                                        instance_type='markers'),
                                           )
                                     )
                marker_props = {Flags.has_markers: True}
                packet.chunks.update({'markers': Chunk(block=marker_block, props=marker_props)})
                if 'events' in packet.chunks:
                    del packet.chunks['events']

        self._data = packet
