import logging
import neuropype.engine as ne


logger = logging.getLogger(__name__)


class FixChannames(ne.Node):
    # --- Input/output ports ---
    data = ne.Port(None, ne.Packet, "Data to process.", required=True,
                   editable=False, mutating=True)

    @classmethod
    def description(cls):
        return ne.Description(name='Rename channels.',
                              description="""
                              analogsignals
                              elec1 --> ch1, etc.
                              """,
                              version='0.1',
                              license=ne.Licenses.MIT)

    @data.setter
    def data(self, pkt):
        if pkt is not None:
            logger.info("Fixing channel names (elecX --> chX) ...")
            blk = pkt.chunks['analogsignals'].block
            new_axes = list(blk.axes)
            sp_ix = blk.axes.index(ne.space)
            new_chan_names = ['ch' + cn[4:] if cn[0:4] == 'elec' else cn for cn in new_axes[sp_ix].names]
            new_axes[sp_ix] = ne.SpaceAxis(names=new_chan_names)
            pkt.chunks['analogsignals'].block = ne.Block(data=blk.data, axes=new_axes)

        self._data = pkt
