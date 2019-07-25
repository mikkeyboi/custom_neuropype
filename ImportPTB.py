import logging
import scipy.io
import numpy as np
from neuropype.engine import *
from neuropype.utilities.cloud import storage


logger = logging.getLogger(__name__)

# Cue Info. The PTB code uses trial classes for which the value indexes into a predefined colour-map, below.
cueColourMap = ['r', 'g', 'b']
cueMatrix = [[1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]]  # 1=red; 2=green; 3=blue. TODO: 0-based.

# Target Info for SR3
# In Adam's code, the 1:8 part of the class is mapped to theta with
# targetTheta = 45*((1:8)-8-1); then the targetPt is calculated as
# target_yx = centre_yx + radius*[-cosd(targetTheta) sind(targetTheta)];
# Notice yx, not xy. Then he fliplr's the result.
# This can be more simply represented the following way.
targetTheta = np.deg2rad([270, 315, 0, 45, 90, 135, 180, 225])
targetStr = ['UU', 'UR', 'RR', 'DR', 'DD', 'DL', 'LL', 'UL']  # The y-zero was up, so flip u/d.

# Only needed for SR4: groups for general rules.
context_options = ['AnyUp', 'AnyDown', 'AnyLeft', 'AnyRight']
# [d;d;d],[u;u;u]
# [l;l;l];[r;r;r]
targGroups = np.array([[[8, 1, 2], [6, 5, 4]], [[8, 7, 6], [2, 3, 4]]])
targDistGroups = np.stack((targGroups, np.flip(targGroups, axis=1)), axis=-1)


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

        # Fix lack of stopTime on last trial --> use last flipScreen.
        if not isinstance(trial_data[-1]['stopTime'], float) or np.isnan(trial_data[-1]['stopTime']):
            trial_data[-1]['stopTime'] = max(trial_data[-1]['flipScreen'])

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
        tr_map = {'ExperimentType': 'expType', 'Block': 'block', 'OutcomeCode': 'outcomeCode',
                  'TargetChoice': 'targetChoice', 'TrialIdx': 'expTrial', 'Class': 'class'}
        check_map = {'SR3errorStrategy': 'SR3errorStrategy', 'SR3trainingLevel': 'SR3trainingLevel',
                     'SR3InitialtrainingLevel': 'SR3InitialtrainingLevel'}
        ev_map = {'Start': 'startTime', 'Cue': 'cuePresentedTime', 'Delay': 'cueExtinguishedTime',
                  'Go': 'fixPointExtinguishedTime', 'Reward': 'rewardTime', 'Stop': 'stopTime'}

        col_names = ['Marker', 'Time', 'EyelinkTime'] + [k for k in tr_map.keys()] + [k for k in check_map.keys()]
        df = pd.DataFrame(columns=col_names)

        for tr_ix, tr_dat in enumerate(trial_data):
            # Collect trial-level variables
            tr = {k: tr_dat[v] for k, v in tr_map.items()}
            tr = {**tr, **{k: tr_dat[v] if (k in tr_dat.dtype.names and isinstance(tr_dat[k], float))
                                            else np.nan for k, v in check_map.items()}}
            trial_t0 = tr_dat['startTime']
            for ev_str, ev_v in ev_map.items():
                # Check why first trial Time is double but rest are OK.
                if isinstance(tr_dat[ev_v], float):  # Is there a faster way to check that it is not empty np.array?
                    ev_t_delta = tr_dat[ev_v] if ev_str != 'Start' else 0
                    ev_dict = {
                        'Marker': ev_str,
                        'Time': trial_t0 + ev_t_delta,
                        'EyelinkTime': tr_dat['eyeSyncTime']/1000. + ev_t_delta
                    }
                    df = df.append({**tr, **ev_dict}, ignore_index=True)

        centrePt = mat['params']['subjectScreenResolution'][()] / 2


        uq_exp, uq_idx = np.unique(df['ExperimentType'], return_inverse=True)
        for _ix, expType in enumerate(uq_exp):
            b_trial = uq_idx == _ix
            classes = df[b_trial]['Class']
            nTrials = len(classes)

            trialTypes = ['unknown'] * nTrials

            # Trial type can be determined based on the experiment type for expTypes M and A.
            """
            if strcmpi(expType, 'M')
                radius = ptbParams.FMgridLength./(2*(ptbParams.FMnumAnnuli:-1:1));
                fixJitterXY = [0 0]; %Add to fix/targ/distr XY
                trialTypes = repmat({'M'}, 1, nTrials);
            elseif strcmpi(expType, 'A')
                radius = ptbParams.targetDistance;  % ??
                trialTypes = repmat({'A'}, 1, nTrials);
            """
            # For expType C or D, we need more information, and that can change on a per-trial basis.
            # Trial type for expType C and D is more complicated.
            # Ultimately it comes down to whether the PTB code calls
            # dSR3T2c_setscreens2 or dSR4T2c_setscreens2. The code path is as
            # follows:
            # (start in saccadeGUImain)
            # if on probation -> doSR3Trial2e
            # if not on probation -> doSR3Trial2d
            #
            # Whether or not we are on probation can be changed throughout the
            # experiment with a key press -> saved to trials.SR3errorStrategy,
            # but only on the trial for which the key was pressed (how can we
            # know what the error strategy was at the beginning of the
            # experiment?!)
            #
            # doSR3Trial2e runs the same trial over and over until 2/3 are
            # successful, then it calls dSR3T2c_setscreens2
            # doSR3Trial2d runs dSR3T2c_setscreens2 if expType C, or
            # dSR4T2c_setscreen2 if expType D.
            #
            # so, expType C is always dSR3T2c_setscreens2

            if expType == 'C':
                radius = mat['params']['SR3radius'][()]
                fixJitterXY = [mat['params'][_][()] for _ in ['SR3Xtranslate', 'SR3Ytranslate']]
                trialTypes = ['SR3'] * nTrials

            elif expType == 'D':
                # TODO: Test indexing [1] for params arrays
                radius = mat['params']['SR3radius'][1]
                fixJitterXY = [mat['params'][_][1] for _ in ['SR3Xtranslate', 'SR3Ytranslate']]
                trialTypes = ['SR4'] * nTrials
                if mat['params']['currSession'][()] <= 49:
                    # JerryLee sra3_2 before 090626 are actually SR3, but every sra3_2
                    # after that is SR4. I haven't yet spotted a sure-fire way to know
                    # that other than from the date.
                    trialTypes = ['SR3'] * nTrials
                else:
                    b_has_strat = ~df['SR3errorStrategy'].isna()
                    print("TODO: Test this")
                    """
                    err_chng_to = [trials(b_has_strat).SR3errorStrategy];
                    err_chng_to = {err_chng_to.error};
                    # if we don't know the beginning error strategy, assume
                    # resample.
                    if ~b_has_strat[0]:
                        b_has_strat[0] = true
                        err_chng_to = ['resample' err_chng_to]

                    is_prob = np.zeros((nTrials,), dtype=bool);
                    err_chng_id = np.where(has_err_strat)[0]
                    for e_ix in range(len(err_chng_id))
                        if e_ix == len(err_chng_id):
                            this_ix = err_chng_id[e_ix]:nTrials;
                        else:
                            this_ix = err_chng_id[e_ix]:err_chng_id(e_ix + 1) - 1

                        is_prob(this_ix) = strcmpi(err_chng_to{e_ix}, 'probation')

                    trialTypes(is_prob) = repmat({'SR3'}, 1, sum(is_prob));
                    """

            # Training levels for each trial.
            # Determines whether target,distractor were used.
            trainingLevels = np.zeros((nTrials,))
            # TODO: Should SR3trainingLevel or SR3InitialtrainingLevel get priority here?
            b_tr = ~df['SR3trainingLevel'].isna()
            trainingLevels[b_tr] = df['SR3trainingLevel'][b_tr]
            b_itr = df['SR3trainingLevel'].isna() & ~df['SR3trainingLevel'].isna()
            trainingLevels[b_itr] = df['SR3InitialtrainingLevel'][b_itr]
            targBool = trainingLevels != 1  # trial has target
            distBool = np.in1d(trainingLevels, [1, 3, 4])  # trial has distractor
            # [trials(trialBool(~distBool)).newType] = deal('CentreOut');

            # Get per-trial indices into targetTheta (targ/dist), cueColourMap, and per trial radii
            #  method depends on trialType

            nExpTrials = np.sum(b_trial)
            targ_ix = np.nan * np.ones((nExpTrials,))     # 1-based
            dist_ix = np.nan * np.ones((nExpTrials,))     # 1-based
            cueCol_ix = np.nan * np.ones((nExpTrials,))   # 1-based
            annulus_ix = np.nan * np.ones((nExpTrials,))  # 1-based
            contexts = [] * nExpTrials

            uq_types, uq_inds = np.unique(trialTypes, return_inverse=True)
            for type_ix, trial_type in enumerate(uq_types):
                b_type = uq_inds == type_ix
                temp = classes[b_type]
                if trial_type == 'M':
                    flip_names = ['fixationOnset', 'targetOnset', 'fixationOffset', 'imperativeCue', 'saccadeEnd',
                                  'targetAcqLost']
                    # It was tough to follow the PTB code, I don't know if the above is correct.
                    # For "M" trials, we are looking for correct saccades typically after flipScreen 2 or 3
                    # ??? Class 1 is up, then around the face of a clock in 45
                    # degree increments until 8. If 9:16 present, those are the same except larger amplitude.
                    targ_ix[b_type] = temp % 8
                    targ_ix[targ_ix == 0] = 8
                    annulus_ix[b_type] = 1 + (temp - targ_ix[b_type]) / 8
                    cueCol_ix[b_type] = 1
                elif trial_type == 'A':
                    logger.warning("Not yet implemented expType A.")
                elif trial_type == 'SR3':
                    flip_names = ['fixationOnset', 'targetOnset', 'cueOnset', 'cueOffset', 'fixationOffset',
                                  'targetAcqLost']
                    # imperativeCue == fixationOffset
                    # trial class = 8*(cueRow-1) + 4*(cueIndex-1) + targetConfig
                    # I rename targetConfig -> theta_targ_ix
                    # cueRow is in 1:length(cueMatrix), chosen rand at the start of each block
                    # targetConfig is in 1:4, representing 4 direction pairs, rand per block
                    # cueIndex is in 1:2, when 1 the targ is in first half, 2: targ in
                    # second half. targets are in [270 315 0 45 90 135 180 225]
                    # targetChoice, in 1:8, = 4*(cueIndex-1) + targetConfig
                    cueRow = np.ceil(temp / 8)

                    temp = temp - 8 * (cueRow - 1)
                    cueIndex = np.ceil(temp / 4)

                    temp = temp - 4 * (cueIndex - 1)
                    targetConfig = temp

                    targ_ix[b_type] = 4 * (cueIndex - 1) + targetConfig
                    dist_ix[b_type] = targ_ix[b_type] + 4 * (targ_ix[b_type] < 5) - 4 * (targ_ix[b_type] > 4)

                    # TODO: cueCol_ix[b_type] = sub2ind(cueMatrix.shape, cueRow, cueIndex)
                    annulus_ix[b_type] = 1
                    contexts[b_type] = targetStr(targ_ix[b_type])
                elif trial_type == 'SR4':
                    flip_names = ['fixationOnset', 'targetOnset', 'cueOnset', 'cueOffset', 'fixationOffset', 'saccadeEnd',
                                  'targetAcqLost']

                    # trial class = 108*(r-1) + 54*(y-1) + 9*(x-1) + z

                    # r is cueColumn; 1 for u|l correct, 2 for d|r corr, rand per trial
                    r = np.ceil(temp / 108)
                    temp = temp - 108 * (r - 1)

                    # y is 1 for u/d, 2 for l/r, chosen randomly per block
                    y = np.ceil(temp/54)
                    temp = temp - 54*(y-1)

                    # x is cueRow in cueMatrix, in 1:6, (determines targ/dist colour pairings per block)
                    x = np.ceil(temp / 9)
                    temp = temp - 9 * (x - 1)

                    # z in 1:9, rand per block. Indexes into target,distractor pairings
                    # t order: 1 2 3 1 2 3 1 2 3 into targDistGroups(t_ix, r, y, 1)
                    # d order: 1 1 1 2 2 2 3 3 3 into targDistGroups(d_ix, r, y, 2)
                    z = temp
                    t_ix = z % 3
                    t_ix[t_ix == 0] = 3
                    d_ix = np.ceil(z/3)

                    targ_ix[b_type] = targDistGroups[
                        np.ravel_multi_index((t_ix, r, y, np.ones(t_ix.shape)), targDistGroups.shape)]
                    dist_ix[b_type] = targDistGroups[
                        np.ravel_multi_index((d_ix, r, y, 2 * np.ones(t_ix.shape)), targDistGroups.shape)]
                    cueCol_ix[b_type] = np.ravel_multi_index((x, r), cueMatrix.shape)
                    annulus_ix[b_type] = np.ones(annulus_ix[b_type].shape)
                    contexts[b_type] = context_options[(y-1)*2 + r]
                else:
                    logger.error("trial type not recognized.")

            """
            %% Save to trial structure
    %.newType = 'M', 'A', 'SR3', or 'SR4'
    %.cueColour = 'r', 'g', or 'b'
    %.trainingLevel = 0: full exp; 1: fix only; 2: fix & dist; 3: full exp &
    %invis dist; 4: 3 + extra cue; 5: 4 + extra cue
    %.targRule = 'UU', 'UR', 'RR', etc. OR 'AnyUp', 'AnyRight', etc.
    %.targClass = in 1:8 for 8 locations, or in 1:16 with two annuli.
    %.targPol = [theta radius]
    %.targXY = [x y] coordinates in screen pixels.
    %(also distClass, distPol, distXY)
    
    [trials(trialBool).newType] = trialTypes{:};
    
    %Cue colours
    trCueColour = cueMatrix(cueCol_ix);
    trCueColour = cueColourMap(trCueColour);
    [trials(trialBool).cueColour] = trCueColour{:};
    
    %Training levels
    trainingLevels = num2cell(trainingLevels);
    [trials(trialBool).trainingLevel] = trainingLevels{:};
    
    %Contexts
    [trials(trialBool).targRule] = contexts{:};
    
    % Save target/distractor to trial structure
    trRadii = radius(annulus_ix);
    
    targDist = {'targ' 'dist'};
    td_ix = [targ_ix dist_ix];
    td_bool = [targBool distBool];
    for td = 1:2 %For targets and distractors.
        this_ix = td_ix(:, td);  % Indices into targetTheta
        this_bool = td_bool(:, td);  % If this trial had a targ/dist
        % stimulus class of 8 (or 16) possible locations
        trClass = nan(nTrials, 1);
        trClass(this_bool) = this_ix(this_bool);
        trClass(annulus_ix > 1) = annulus_ix(annulus_ix > 1).*trClass(annulus_ix > 1);
        trClass = num2cell(trClass);
        [trials(trialBool).([targDist{td} 'Class'])] = trClass{:};
        % Angle - needed for below
        trTheta = nan(nTrials, 1);
        trTheta(this_bool) = targetTheta(this_ix(this_bool));
        % Pol, assuming centre = 0, 0
        trPol = num2cell([trTheta trRadii], 2);
        [trials(trialBool).([targDist{td} 'Pol'])] = trPol{:};
        % Coordinates in screen pixels
        [trX, trY] = pol2cart(trTheta, trRadii);
        trXY = round(bsxfun(@plus, centrePt + fixJitterXY, [trX trY]));
        trXY = num2cell(trXY, 2);
        [trials(trialBool).([targDist{td} 'XY'])] = trXY{:};
        %String representing direction.
        trStr = cell(nTrials, 1);
        trStr(this_bool) = targetStr(this_ix(this_bool));
        [trials(trialBool).([targDist{td} 'Str'])] = trStr{:};
    end
            """


        iax = InstanceAxis(df['Time'].values, data=df.drop(columns=['Time']).to_records(index=False))
        ev_blk = Block(data=np.nan * np.ones((len(iax),)), axes=(iax,))
        ev_props = {Flags.is_event_stream: True,
                    'gaze_calib_adjust': (gaze_calib_adjust,),  # Need to hide in tuple so it doesn't break enumerator
                    'ptb_params': {k: mat['params'][k][()] for k in mat['params'].dtype.names}
                    }

        self._data = Packet(chunks={'markers': Chunk(block=ev_blk, props=ev_props)})
