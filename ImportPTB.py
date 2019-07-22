import logging
import scipy.io
import numpy as np
from neuropype.engine import *
from neuropype.utilities.cloud import storage


logger = logging.getLogger(__name__)


class ImportPTB(Node):
    # --- Input/output ports ---
    filename = StringPort("", """Name of the event dataset.
                    """, is_filename=True, direction=IN)
    data = Port(None, Packet, "Output with markers chunk", required=True, direction=OUT)

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
        return Description(name='Import PyschoPhysicsToolbox data file'
                                ' from macaque saccade experiments',
                           description="""
                           """,
                           version='0.1',
                           license=Licenses.MIT)

    @Node.update.setter
    def update(self, v):

        filename = storage.cloud_get(self.filename, host=self.cloud_host,
                                     account=self.cloud_account,
                                     bucket=self.cloud_bucket,
                                     credentials=self.cloud_credentials)

        logger.info("Loading Psychophysicstoolbox events from %s..." % filename)
        mat = scipy.io.loadmat(filename, squeeze_me=True)
        trial_data = mat['eData']['trial'][()]  # recarray

        # Do I need mat['DIO'].dtype.names? So far, no.

        # Save online eyetracker calibration adjustments
        if 'CALADJ' in mat:
            gaze_calib_adjust = mat['CALADJ']
        elif 'eyePosCalibAdjust' in mat['eData'].dtype.names:
            gaze_calib_adjust = mat['eData']['eyePosCalibAdjust'][()]
        else:
            gaze_calib_adjust = None

        # Normalize field names that vary across data sets.
        import numpy.lib.recfunctions as rfn
        trial_data = rfn.rename_fields(trial_data, {
            'trialStartTime': 'startTime',
            'trialStopTime': 'stopTime',
            'eyeSyncStartTime': 'eyeSyncTime',
            'trialID': 'ID',
            'timeofrelease': 'leverRelease',
            'cueTime': 'cuePresentedTime'
        })

        # TODO: Fix lack of stopTime on last trial --> nan
        # trial_data[-1]['stopTime']

        # If needed, replace stopTime with trial duration
        if (trial_data[0]['stopTime'] - trial_data[0]['startTime']) > 0:
            trial_data['stopTime'] = trial_data['stopTime'] - trial_data['startTime']

        # I'm not sure about this next one...
        # In files with both M and A trial types,
        # M trials have targetChoice 1-8 (maybe 1-16) and class is empty.
        # A trials have targetChoice 0 or 1 and class is targetChoice+1.
        if len(np.unique(trial_data['expType'])) > 1:
            trial_data = rfn.drop_fields(trial_data, ['class'])
            trial_data = rfn.rename_fields(trial_data, {'targetChoice': 'class'})

        # Create an event stream chunk using each important event in each trial
        import pandas as pd
        df = pd.DataFrame(columns=['Marker', 'Time', 'EyelinkTime', 'ExperimentType',
                                   'Block', 'OutcomeCode', 'TargetChoice', 'TrialIdx',
                                   'Class'])
        ev_map = [('Start', 'startTime'), ('Cue', 'cuePresentedTime'), ('Delay', 'cueExtinguishedTime'),
                  ('Go', 'fixPointExtinguishedTime'), ('Reward', 'rewardTime'), ('Stop', 'stopTime')]
        for tr_ix, tr_dat in enumerate(trial_data):
            # Collect trial-level variables
            tr = {'ExperimentType': tr_dat['expType'], 'Block': tr_dat['block'], 'OutcomeCode': tr_dat['outcomeCode'],
                  'TargetChoice': tr_dat['targetChoice'], 'TrialIdx': tr_dat['expTrial'], 'Class': tr_dat['class']}
            t0 = tr_dat['startTime']
            for ev in ev_map:
                # Check why first trial Time is double but rest are OK.
                if isinstance(tr_dat[ev[1]], float):  # Is there a faster way to check that it is not empty np.array?
                    t_delta = tr_dat[ev[1]] if ev[0] != 'Start' else 0
                    ev_dict = {
                        'Marker': ev[0],
                        'Time': t0 + t_delta,
                        'EyelinkTime': tr_dat['eyeSyncTime']/1000 + t_delta
                    }
                    df = df.append({**tr, **ev_dict}, ignore_index=True)

        iax = InstanceAxis(df['Time'].values, data=df.drop(columns=['Time']).to_records(index=False))
        ev_blk = Block(data=np.nan * np.ones((len(iax),)), axes=(iax,))
        ev_props = {Flags.is_event_stream: True,
                    'gaze_calib_adjust': (gaze_calib_adjust,),  # Need to hide in tuple so it doesn't break enumerator
                    'ptb_params': {k: mat['params'][k][()] for k in mat['params'].dtype.names}
                    }
        self._data = Packet(chunks={'markers': Chunk(block=ev_blk, props=ev_props)})
