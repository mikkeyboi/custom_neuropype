import logging
import numpy as np
from neuropype.engine import *

logger = logging.getLogger(__name__)


class GetUnityTaskEvents(Node):
    # --- Input/output ports ---
    data = Port(None, Packet, "Data to process.", required=True,
                editable=False, mutating=True)

    @classmethod
    def description(cls):
        return Description(name='Get Behaviour for Michael Saccade VR study',
                           description="""Parse marker strings into table of data""",
                           version='0.1',
                           license=Licenses.MIT)

    @data.setter
    def data(self, pkt):
        mrk_n, mrk_chnk = find_first_chunk(pkt, name_equals='markers')
        if mrk_n is not None:
            ev_times = mrk_chnk.block.axes[instance].times

            # Load the data
            import json
            dict_arr = mrk_chnk.block.axes[instance].data['Marker']
            events = []
            for ix, ev in enumerate(dict_arr):
                # Fix some mistakes in the json encoding in Unity
                dat = json.loads(ev)
                if 'CameraRecenter:' in dat:
                    dat = {'CameraRecenter': dat['CameraRecenter:']}
                if 'Input:' in dat:
                    dat = {'Input': dat['Input:']}
                events.append(dat)

            """
            Input event markers:
                TrialState:
                    isCorrect (bool)
                    trialIndex (uint)
                    taskType (int): See tasktype_map
                    inhibitionIndex: See countermand_map
                    cuedPositionIndex: See position_map
                    targetPositionIndex: See position_map
                    targetObjectIndex (int): Different stimuli. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
                    selectedObjectIndex (int): in -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9
                    selectedPositionIndex: in -1, 0, 1
                    environmentIndex: 0
                    targetColorIndex: 3
                    trialPhaseIndex: see phase_map
                Input: An event whenever a user input is registered (e.g., gaze collides with object)
                    trialIndex (int)
                    selectedObjectClass (str): in 'Background', 'Fixation', 'Target', 'Wall'
                    info (key-value pair): 'Selected: <selected object name>'
                ObjectInfo:
                    _isVisible (bool)
                    _identity (string)
                    _position (x,y,z)
                    _pointingTo (x,y,z)
                CameraRecenter: (bool) Camera height and yaw recentered on user
            """
            # Trial phase indices map to trial phases
            phase_map = {1: 'Intertrial', 2: 'Fixate', 3: 'Cue', 4: 'Delay', 5: 'Target',
                         6: 'Go', 7: 'Countermand', 8: 'Response', 9: 'Feedback', -1: 'UserInput'}
            phase_inv_map = {v: k for k, v in phase_map.items()}
            tasktype_map = {0: 'AttendShape', 1: 'AttendColour', 2: 'AttendDirection'}
            countermand_map = {0: 'Prosaccade', 1: 'TargetSwitch', 2: 'NoGo', 3: 'Antisaccade'}
            position_map = {-1: 'Unknown', 0: 'Left', 1: 'Right', 2: 'NoGo'}

            """
            There are many more events than we need, including events for positioning invisible targets and changing
            their colour.
            For each trial, we want to keep any events where the stimulus changed or where the user saw something.
            Each row will also have other data that describe the whole trial, so when we select individual events
            later, we still have all of the info we need to know what kind of trial it was.
            Note that the ObjectInfo events occur before their associated TrialState event, so the most accurate
            timestamps will come from ObjectInfo, not TrialState.
            
            Trial lifecycle:
            - ObjectInfo event when target is placed but still invisible
            - TrialState event with trialPhaseIndex 1 to indicate intertrial
            - Input event (>=1) to indicate subject is selecting CentralFixation / CentralWall.
            - TrialState with trialPhaseIndex = 2 to indicate Fixate phase.
            - Last Input event must be CentralFixation to proceed.
            - ObjectInfo to show the cue. (_isVisible: True)
            - TrialState with trialPhaseIndex=3 to indicate cue phase.
            - multiple ObjectInfo events with _isVisible False while the cue disappears and targets are positioned.
            - TrialState with trialPhaseIndex=4 for the Delay (memory) period.
            - ObjectInfo events to make the targets visible.
            - TrialState event with trialPhaseIndex 5 to indicate this is the target phase (map memory to saccade plan)
            - ObjectInfo with CentralFixation set to _isVisible False. This is the imperative go cue.
            - TrialState with trialPhaseIndex 6 to indicate the Go phase. TODO: Check if the time is same as above.
            - (Optional) Input event after fixation disappears because we are now selecting CentralWall behind fixation.
            - (if countermanding) ObjectInfo when fixation reappears. Start of countermanding.
            - (if countermanding) Input when fixation goes back on to central
            - TrialState with trialPhaseIndex 7 to indicate beginning of countermanding phase, whether or not stim given
            - ObjectInfo when CentralFixation disappears again
            - TrialState with trialPhaseIndex 8 to indicate beginning of Response phase
            - Input to indicate hitting target (or non-target, or opposite wall in antisaccade)
            - ObjectInfo to clear out CentralFixation
            - TrialState with trialPhaseIndex 8 again, but this time the isCorrect has changed.
            - TrialState with trialPhaseIndex 9 to indicate feedback phase.
            The next ObjectInfo event indicates the start of the next trial
            """

            # Output table will have the following fields
            field_name_type_prop = [
                ('UnityTrialIndex', int, ValueProperty.INTEGER + ValueProperty.NONNEGATIVE),
                ('Marker', object, ValueProperty.STRING + ValueProperty.CATEGORY),  # Used to hold trial phase.
                ('TaskType', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('CountermandingType', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('CuedPosition', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('CuedObject', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('TargetPosition', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('TargetObjectIndex', int, ValueProperty.INTEGER + ValueProperty.CATEGORY),
                # ('TargetColour', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('EnvironmentIndex', int, ValueProperty.INTEGER + ValueProperty.NONNEGATIVE),
                ('CountermandingDelay', float, ValueProperty.UNKNOWN),
                ('SelectedPosition', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('SelectedObjectIndex', int, ValueProperty.INTEGER + ValueProperty.CATEGORY),
                ('IsCorrect', bool, ValueProperty.NONNEGATIVE),
                ('ReactionTime', float, ValueProperty.UNKNOWN),
            ]
            field_names, field_types, field_props = zip(*field_name_type_prop)
            ra_dtype = list(zip(zip(field_props, field_names), field_types))  # For recarray

            # Identify the trial index for each event, even the ObjectInfo and Input events.
            ev_types = np.array([list(_.keys())[0] for _ in events])
            last_tr_ind = 0
            last_phase = 9
            object_bump = False
            ev_tr = []
            for ev_ix, ev in enumerate(events):
                if ev_types[ev_ix] == 'TrialState':
                    last_phase = ev['TrialState']['trialPhaseIndex']
                    last_tr_ind = ev['TrialState']['trialIndex']
                    object_bump = False
                elif ev_types[ev_ix] == 'ObjectInfo' and last_phase == 9 and not object_bump:
                    # The first ObjectInfo event after a phase-9 event is the start of a new trial.
                    last_tr_ind += 1
                    object_bump = True
                ev_tr.append(last_tr_ind)
            ev_tr = np.array(ev_tr)
            
            # ev_tr might wrap if there were multiple files loaded.
            while np.any(np.diff(ev_tr) < 0):
                switch_ind = np.where(np.diff(ev_tr) < 0)[0] + 1
                offset = ev_tr[switch_ind - 1]
                ev_tr[switch_ind[0]:] += offset
            
            # Start to build the dataframe
            import pandas as pd
            df = pd.DataFrame(columns=field_names)
            out_times = []

            for tr_ix, tr_ind in enumerate(np.unique(ev_tr)):
                b_tr = ev_tr == tr_ind
                tr_types = ev_types[b_tr]
                if 'TrialState' not in tr_types:
                    continue
                tr_times = ev_times[b_tr]
                tr_events = np.array(events)[b_tr]

                tr_phases = np.array([_['TrialState']['trialPhaseIndex']
                                      if 'TrialState' in _ else np.nan for _ in tr_events])

                # Details to be saved along with each event for this trial.
                # Every trial should have feedback phase and it should be the most informative.
                if phase_inv_map['Feedback'] not in tr_phases:
                    continue
                fbstate = tr_events[tr_phases == phase_inv_map['Feedback']][0]['TrialState']
                details = {
                    'UnityTrialIndex': fbstate['trialIndex'],
                    'TaskType': tasktype_map[fbstate['taskType']],
                    'CountermandingType': countermand_map[fbstate['inhibitionIndex']],
                    'CuedPosition': position_map[fbstate['cuedPositionIndex']],
                    'TargetPosition': position_map[fbstate['targetPositionIndex']],
                    'TargetObjectIndex': fbstate['targetObjectIndex'],  # TODO: Map to object name
                    # 'TargetColour': color_map[fbstate['targetColorIndex']],
                    'EnvironmentIndex': fbstate['environmentIndex'],    # TODO: Map to environment name.
                    'SelectedPosition': position_map[fbstate['selectedPositionIndex']],
                    'SelectedObjectIndex': fbstate['selectedObjectIndex'],  # TODO: Map to object name
                    'IsCorrect': fbstate['isCorrect'],
                    'CountermandingDelay': np.nan,
                    'ReactionTime': np.nan
                }
                # Get some more details that we can only get from other events.

                # 'CuedObject'
                if phase_inv_map['Cue'] in tr_phases:
                    ph_ix = np.where(tr_phases == phase_inv_map['Cue'])[0][0]
                    cue_ix = np.where(tr_types[:ph_ix] == 'ObjectInfo')[0][-1]
                    details['CuedObject'] = tr_events[cue_ix]['ObjectInfo']['_identity']
                else:
                    cue_ix = None

                # Get go time
                if phase_inv_map['Go'] in tr_phases:
                    ph_ix = np.where(tr_phases == phase_inv_map['Go'])[0][0]
                    obj_ix = np.where(tr_types[:ph_ix] == 'ObjectInfo')[0][-1]
                    obj_inf = tr_events[obj_ix]['ObjectInfo']
                    go_ix = obj_ix if (obj_inf['_identity'] == 'CentralFixation') and (not obj_inf['_isVisible'])\
                        else ph_ix
                    go_time = tr_times[go_ix]
                else:
                    go_ix = None

                # Get countermanding delay
                if details['CountermandingType'] != 'Prosaccade' and phase_inv_map['Countermand'] in tr_phases:
                    ph_ix = np.where(tr_phases == phase_inv_map['Countermand'])[0][0]
                    obj_ix = ph_ix + np.where(tr_types[ph_ix:] == 'ObjectInfo')[0][0]
                    obj_inf = tr_events[obj_ix]['ObjectInfo']
                    cm_ix = obj_ix if (obj_inf['_identity'] == 'CentralFixation') and obj_inf['_isVisible'] else ph_ix
                    details['CountermandingDelay'] = tr_times[cm_ix] - go_time
                else:
                    cm_ix = None

                # Get reaction time
                if phase_inv_map['Response'] in tr_phases:
                    ph_ix = np.where(tr_phases == phase_inv_map['Response'])[0][0]
                    resp_ix = ph_ix + np.where(tr_types[ph_ix:] == 'Input')[0]
                    resp_ix = [_ for _ in resp_ix if tr_events[_]['Input']['selectedObjectClass'] != 'Fixation']
                    resp_ix = resp_ix[0] if len(resp_ix) > 0 else ph_ix
                    details['ReactionTime'] = tr_times[resp_ix] - go_time
                else:
                    resp_ix = None

                # Append timestamp and details for each interesting event.
                # Event 1 - Intertrial. ObjectInfo cue placed but hidden. Use phase transition.
                ph_ix = np.where(tr_phases == phase_inv_map['Intertrial'])[0][0]
                df = df.append({'Marker': 'Intertrial', **details}, ignore_index=True)
                out_times.append(tr_times[ph_ix])

                # Event 2 - Fixation achieved. Use phase transition.
                if phase_inv_map['Fixate'] in tr_phases:
                    fix_start_ix = np.where(tr_phases == phase_inv_map['Fixate'])[0][0]
                    df = df.append({'Marker': 'Fixate', **details}, ignore_index=True)
                    out_times.append(tr_times[fix_start_ix])

                # Event 3 - Cue presentation. Object info cue appears, transition to phase 3.
                # Already found above while getting cued object.
                if cue_ix is not None:
                    df = df.append({'Marker': 'Cue', **details}, ignore_index=True)
                    out_times.append(tr_times[cue_ix])

                # Event 4 - Delay period. ObjectInfo cue disappears; transition to phase 4.
                if phase_inv_map['Delay'] in tr_phases:
                    pre_ix = np.where(tr_phases == phase_inv_map['Cue'])[0][0]
                    ph_ix = np.where(tr_phases == phase_inv_map['Delay'])[0][0]
                    del_ix = pre_ix + np.where(tr_types[pre_ix:ph_ix] == 'ObjectInfo')[0][0]
                    df = df.append({'Marker': 'Delay', **details}, ignore_index=True)
                    out_times.append(tr_times[del_ix])

                # Event 5 - Target presentation. ObjectInfo targets appear; transition to phase 5.
                if phase_inv_map['Target'] in tr_phases:
                    ph_ix = np.where(tr_phases == phase_inv_map['Target'])[0][0]
                    targ_ix = np.where(tr_types[:ph_ix] == 'ObjectInfo')[0][0]
                    df = df.append({'Marker': 'Target', **details}, ignore_index=True)
                    out_times.append(tr_times[targ_ix])

                # Event 6 - Imperative cue. Fixation pt disappears. Transition to Phase 6.
                # Already found above.
                if go_ix is not None:
                    df = df.append({'Marker': 'Go', **details}, ignore_index=True)
                    out_times.append(tr_times[go_ix])
                else:
                    logger.debug("Go cue not found for trial {}.".format(tr_ind))

                # Event 7 (optional) - Countermanding cue.
                # Already found above.
                if cm_ix is not None:
                    df = df.append({'Marker': 'Countermand', **details}, ignore_index=True)
                    out_times.append(tr_times[cm_ix])

                # Event 8 - Response. Without pupil data yet, we use Input event.
                # Already found above.
                if resp_ix is not None:
                    df = df.append({'Marker': 'Response', **details}, ignore_index=True)
                    out_times.append(tr_times[resp_ix])

                # Event 9 - Feedback. Use phase transition.
                ph_ix = np.where(tr_phases == phase_inv_map['Feedback'])[0][0]
                df = df.append({'Marker': 'Feedback', **details}, ignore_index=True)
                out_times.append(tr_times[ph_ix])

            # Try to infer column datatypes.
            # df.infer_objects()  <- requires pandas >= 0.21
            df['UnityTrialIndex'] = df['UnityTrialIndex'].astype(int)

            # Modify instance axis
            new_data = df.to_records(index=False).astype(ra_dtype)
            pkt.chunks[mrk_n].block = Block(data=np.full((len(out_times),), np.nan),
                                            axes=(InstanceAxis(times=out_times, data=new_data,
                                                               instance_type='markers'),))

        self._data = pkt
