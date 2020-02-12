import logging
import numpy as np
import copy
from neuropype.engine import *

logger = logging.getLogger(__name__)


class GetUnityTaskEvents(Node):
    # --- Input/output ports ---
    data = Port(None, Packet, "Data to process.", required=True,
                editable=False, mutating=True)

    @classmethod
    def description(cls):
        return Description(name='Get Behaviour for Michael Saccade VR study (New submodule post Sept 10 revisions)',
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
                    condition (int): See conditiontype_map
                    isCorrect (bool)
                    modifier (int): See modifiertype_map
                    trialIndex (uint)
                    response: See responsetype_map
                    cuedPositionIndex: See position_map
                    targetPositionIndex: See position_map
                    targetObjectIndex (int): 0
                    selectedObjectIndex (int): in -1, 0
                    selectedPositionIndex: in -1, 0, 1
                    targetColorIndex: -1
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
            phase_map = {0: 'Setup', 1: 'Intertrial', 2: 'Fixation', 3: 'Cue', 4: 'Delay_1', 5: 'Target',
                         6: 'Delay_2', 7: 'Distractor', 8: 'Delay_3', 9: 'Countermanding', 10: 'Delay_4', 11: 'Response',
                         12: 'Feedback', 13: 'Delay_5', 14: 'Misc', 15: 'Selector', 16: 'Null', -1: 'UserInput'}
            phase_inv_map = {v: k for k, v in phase_map.items()}
            modifiertype_map = {0: 'None', 1: 'Cued', 2: 'MemoryGuided', 3: 'NoGo', 4: 'Catch'}
            conditiontype_map = {0: 'None', 1:'AttendShape', 2: 'AttendColour', 3: 'AttendNumber', 4: 'AttendDirection', 5: 'AttendPosition', 6: 'AttendFixation'}
            # Note: ResponseType 3 to 5 are added post data collection to expedite analysis
            responsetype_map = {0: 'None', 1: 'Prosaccade', 2: 'Antisaccade', 3: 'CuedSaccade', 4: 'NoGoProsaccade', 5: 'NoGoAntisaccade'}
            position_map = {-1: 'Unknown', 0: 'Left', 1: 'Right', 2: 'NoGo'}

            """
            There are many more events than we need, including events for positioning invisible targets and changing
            their colour.
            For each trial, we want to keep any events where the stimulus changed or where the user saw something.
            Each row will also have other data that describe the whole trial, so when we select individual events
            later, we still have all of the info we need to know what kind of trial it was.
            Note that the ObjectInfo events occur before their associated TrialState event, so the most accurate
            timestamps will come from ObjectInfo, not TrialState.
            
            Trial life cycle (Shortest 15 Events):
            1- ObjectInfo event when target is placed but still invisible
            2- TrialState event with trialPhaseIndex 1 to indicates intertrial
             - Input event (>=1) to indicate subject is selecting CentralFixation / CentralWall.
            3- TrialState with trialPhaseIndex = 2 to indicate Fixation phase.
            4- ObjectInfo to show the cue. (_isVisible: True)
            5- TrialState with trialPhaseIndex=3 to indicate cue phase.
            6- ObjectInfo shows colour change of cue to indicate Prosaccade/Antisaccade trial.
            <Additional ObjectInfo to show target in Cued trials>
            7- Input event selects fixation point (!! TODO: This should be during the gating phase)
            8- TrialState event with trialPhaseIndex=4 for the Delay (memory) period.
            9- TrialState event with trialPhaseIndex=6 to indicate this is the variable delay phase (map memory to saccade plan)
            10- TrialState event with trialPhaseIndex=15 to select whether the trial is countermanding or not
            11- ObjectInfo events: _isVisible: True for showing _identity: Target (For Cued Trials, this happens during Cue Phase)
            12- ObjectInfo events: _isVisible: False indicates GO imperative
            13- TrialState with trialPhaseIndex=11 to indicate the Go phase. TODO: Check if the time is same as above.
            14- Input event selects the target
            15- TrialState event with trialPhaseIndex=12 to indicate feedback phase (!! IsCorrect changes here !!)
            
            Countermanding trial life cycle (~17 Events)
            1-12 are the same as the above
            13- TrialState event with trialPhaseIndex=9 to indicate countermand
            14- ObjectInfo event when fixation reappear. Start of countermanding.
            15- TrialState with trialPhaseIndex=11 to indicate Response phase. This is when the subject has to maintain fixation
            16- TrialState event with trialPhaseIndex=12 to indicate feedback phase (!! IsCorrect changes here !!)
            17- ObjectInfo (start of new trial but seems like after countermanding there's an additional ObjectInfo to hide the fixation)
            """

            # Output table will have the following fields
            field_name_type_prop = [
                ('UnityTrialIndex', int, ValueProperty.INTEGER + ValueProperty.NONNEGATIVE),
                ('Marker', object, ValueProperty.STRING + ValueProperty.CATEGORY),  # Used to hold trial phase.
                ('ModifierType', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('ConditionType', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('ResponseType', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('CuedPosition', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('CuedObject', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('TargetPosition', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('TargetObjectIndex', int, ValueProperty.INTEGER + ValueProperty.CATEGORY),
                # ('TargetColour', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                # ('EnvironmentIndex', int, ValueProperty.INTEGER + ValueProperty.NONNEGATIVE),
                ('CountermandingDelay', float, ValueProperty.UNKNOWN),
                ('SelectedPosition', object, ValueProperty.STRING + ValueProperty.CATEGORY),
                ('SelectedObjectIndex', int, ValueProperty.INTEGER + ValueProperty.CATEGORY),
                ('IsCorrect', bool, ValueProperty.NONNEGATIVE),
                ('ReactionTime', float, ValueProperty.UNKNOWN),
                ('CueTypeIndex', object, ValueProperty.STRING + ValueProperty.CATEGORY),
            ]
            field_names, field_types, field_props = zip(*field_name_type_prop)
            ra_dtype = list(zip(zip(field_props, field_names), field_types))  # For recarray

            # Identify the trial index for each event, even the ObjectInfo and Input events.
            ev_types = np.array([list(_.keys())[0] for _ in events])
            last_tr_ind = 0
            last_phase = 12
            object_bump = False
            ev_tr = []
            for ev_ix, ev in enumerate(events):
                if ev_types[ev_ix] == 'TrialState':
                    last_phase = ev['TrialState']['trialPhaseIndex']
                    last_tr_ind = ev['TrialState']['trialIndex']
                    object_bump = False
                elif ev_types[ev_ix] == 'ObjectInfo' and last_phase == 12 and not object_bump:
                    # The first ObjectInfo event after a phase-12 event is the start of a new trial.
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
                    'ModifierType': modifiertype_map[fbstate['modifier']],
                    'ConditionType': conditiontype_map[fbstate['condition']],
                    'ResponseType': responsetype_map[fbstate['response']] if fbstate['modifier'] == 0 else 'CuedOrNoGo',
                    'CuedPosition': position_map[fbstate['cuePositionIndex']],
                    'TargetPosition': position_map[fbstate['targetPositionIndex']],
                    'TargetObjectIndex': fbstate['targetObjectIndex'],  # TODO: Map to object name
                    # 'TargetColour': color_map[fbstate['targetColorIndex']],
                    # 'EnvironmentIndex': fbstate['environmentIndex'],    # TODO: Map to environment name.
                    'SelectedPosition': position_map[fbstate['selectedPositionIndex']],
                    'SelectedObjectIndex': fbstate['selectedObjectIndex'],  # TODO: Map to object name
                    'IsCorrect': fbstate['isCorrect'],
                    'CountermandingDelay': np.nan,
                    'ReactionTime': np.nan
                    # No need for CueTypeIndex, ResponseType indicates whether trial is Pro or Anti-saccade
                    # CueTypeIndex. For "TaskSwitch" experiment, tells if trial is Pro or Anti-saccade.
                    # 'CueTypeIndex': cue_type_map[fbstate['saccadeIndex']] if 'saccadeIndex' in fbstate else -1
                }
                # Additional ResponseTypes are added in analysis (here), for comparing against conditions
                if details['ResponseType'] == 'CuedOrNoGo':
                    details['ResponseType'] = 'CuedSaccade' if fbstate['modifier'] == 1 \
                        else ('NoGoProsaccade' if fbstate['response'] == 1 else 'NoGoAntisaccade')

                # Get some more details that we can only get from events.
                df_to_extend = []
                tr_times = ev_times[b_tr]
                tr_is_obj = tr_types == 'ObjectInfo'
                tr_obj_is_vis = np.array([tr_events[_]['ObjectInfo']['_isVisible'] if tr_is_obj[_] else False
                                          for _ in range(len(tr_events))])
                tr_obj_id = np.array([tr_events[_]['ObjectInfo']['_identity'] if tr_is_obj[_] else None
                                      for _ in range(len(tr_events))])

                # Event 1 - Intertrial. ObjectInfo cue placed but hidden. Use phase transition.
                df_to_extend.append({'Marker': 'Intertrial'})
                iti_ix = np.where(tr_phases == phase_inv_map['Intertrial'])[0][0]
                out_times.append(tr_times[iti_ix])

                # Event 2 - Fixation achieved. Use phase transition.
                if phase_inv_map['Fixation'] in tr_phases:
                    df_to_extend.append({'Marker': 'Fixation'})
                    fix_start_ix = np.where(tr_phases == phase_inv_map['Fixation'])[0][0]
                    out_times.append(tr_times[fix_start_ix])

                # Event 3 - Cue presentation. Transition to phaseIndex 3 and Object appears (maybe reversed order)
                if phase_inv_map['Cue'] in tr_phases:
                    df_to_extend.append({'Marker': 'Cue'})
                    cue_ix = np.where(tr_phases == phase_inv_map['Cue'])[0][0]
                    # TODO: Current experiment does not have a ObjectInfo event near time of cue.
                    # obj_ix = np.where(tr_obj_is_vis)[0][np.argmin(np.abs(tr_times[tr_obj_is_vis] - tr_times[cue_ix]))]
                    # details['CuedObject'] = tr_obj_id[obj_ix]
                    out_times.append(tr_times[cue_ix])  # TODO: use obj_ix in new experiment.

                # Event 4 - Delay period. ObjectInfo cue disappears; transition to phaseIndex 4.
                if phase_inv_map['Delay_1'] in tr_phases:
                    df_to_extend.append({'Marker': 'Delay_1'})
                    pre_ix = np.where(tr_phases == phase_inv_map['Cue'])[0][0]
                    ph_ix = np.where(tr_phases == phase_inv_map['Delay_1'])[0][0]
                    del_ix = pre_ix + np.where(tr_is_obj[pre_ix:ph_ix])[0][0]
                    out_times.append(tr_times[del_ix])

                # Event 5 - Target presentation. ObjectInfo targets appear; transition to phaseIndex 6.
                if phase_inv_map['Delay_2'] in tr_phases:
                    df_to_extend.append({'Marker': 'Delay_2'})
                    targ_ix = np.where(np.logical_and(tr_obj_id == 'Delay_2', tr_obj_is_vis))[0]
                    if len(targ_ix) > 0:
                        targ_ix = targ_ix[-1]
                    else:
                        targ_ix = np.where(tr_phases == phase_inv_map['Delay_2'])[0][0]
                    out_times.append(tr_times[targ_ix])

                # Event 6 - Imperative cue. Fixation pt disappears. Transition to phaseIndex 15.
                if phase_inv_map['Selector'] in tr_phases:
                    df_to_extend.append({'Marker': 'Go'})
                    go_ix = np.where(np.logical_and(tr_obj_id == 'CentralFixation', ~tr_obj_is_vis))[0]
                    if len(go_ix) > 0:
                        go_ix = go_ix[0]
                    else:
                        go_ix = np.where(tr_phases == phase_inv_map['Selector'])[0][0]
                    go_time = tr_times[go_ix]
                    out_times.append(go_time)
                else:
                    logger.debug("Go cue not found for trial {}.".format(tr_ind))

                # Event 7 (optional) - Countermanding cue.
                # Get countermanding delay
                if details['ResponseType'] != 'Prosaccade' and phase_inv_map['Countermanding'] in tr_phases:
                    df_to_extend.append({'Marker': 'Countermand'})

                    # Find last fixation-visible event before response period.
                    resp_ix = np.where(tr_phases == phase_inv_map['Response'])[0][0]
                    b_countermand = np.logical_and(tr_obj_id[:resp_ix] == 'CentralFixation', tr_obj_is_vis[:resp_ix])
                    cm_ix = np.where(b_countermand)[0]
                    if len(cm_ix) > 0:
                        cm_ix = cm_ix[-1]
                    else:
                        cm_ix = np.where(tr_phases == phase_inv_map['Countermanding'])[0][0]
                    details['CountermandingDelay'] = tr_times[cm_ix] - go_time
                    out_times.append(tr_times[cm_ix])

                # Event 8 - Response. Without pupil data yet, we use Input event.
                # Get reaction time
                if phase_inv_map['Response'] in tr_phases:
                    df_to_extend.append({'Marker': 'Response'})
                    ph_ix = np.where(tr_phases == phase_inv_map['Response'])[0][0]
                    resp_ix = ph_ix + np.where(tr_types[ph_ix:] == 'Input')[0]
                    resp_ix = [_ for _ in resp_ix if tr_events[_]['Input']['selectedObjectClass'] != 'Fixation']
                    resp_ix = resp_ix[0] if len(resp_ix) > 0 else ph_ix
                    details['ReactionTime'] = tr_times[resp_ix] - go_time
                    out_times.append(tr_times[resp_ix])

                # Event 9 - Feedback. Use phase transition.
                df_to_extend.append({'Marker': 'Feedback'})
                fb_ix = np.where(tr_phases == phase_inv_map['Feedback'])[0][0]
                out_times.append(tr_times[fb_ix])

                for new_ev in df_to_extend:
                    df = df.append(dict(new_ev, **details), ignore_index=True)

            # Try to infer column datatypes.
            # df.infer_objects()  <- requires pandas >= 0.21
            df['UnityTrialIndex'] = df['UnityTrialIndex'].astype(int)

            # Modify instance axis
            new_data = df.to_records(index=False).astype(ra_dtype)
            pkt.chunks[mrk_n].block = Block(data=np.full((len(out_times),), np.nan),
                                            axes=(InstanceAxis(times=out_times, data=new_data,
                                                               instance_type='markers'),))

        self._data = pkt
