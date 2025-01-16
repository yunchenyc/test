#%% Import
from pynwb import NWBHDF5IO
import os
import json
import numpy as np


#%% Check version
# import pynwb
# import h5py
# import hdmf

# print(pynwb.__version__)
# print(h5py.__version__)
# print(hdmf.__version__)

#%% Load nwb file
dirpath = format('F://Aac//Work//Project//Interception//Data//abel_20240501')
# dirpath = format('F://Aac//Work//Project//Interception//Data')

# # integrated data
filepath = os.path.join(dirpath, 'standard_data.nwb')
io = NWBHDF5IO(filepath, mode='r')
nwbfile = io.read()
print(nwbfile)


#%% Check time difference
# # bhv data
# filepath = os.path.join(dirpath, 'continuous_behavior.nwb')
# io = NWBHDF5IO(filepath, mode='r')
# bhvfile = io.read()
# print(bhvfile)

# # bhv - event_time
# bhv_code = bhvfile.processing['behavior']['MonkeyLogicEvents'].data[:]
# bhv_time = bhvfile.processing['behavior']['MonkeyLogicEvents'].timestamps[:]

# # bhv - code_time
# bhv_df = bhvfile.trials.to_dataframe()
# bhv_trial_time = [i for _, trial in bhv_df.iterrows() for i in json.loads(trial['BehavioralCodes_CodeTimes'])]

# # bhv - abs_code_time
# bhv_time2 = [i+trial['AbsoluteTrialStartTime'] 
#              for _, trial in bhv_df.iterrows() for i in json.loads(trial['BehavioralCodes_CodeTimes'])]

# # bhv - abs_time
# abs_time = [i for _, trial in bhv_df.iterrows() 
#             for i in np.full(len(json.loads(trial['BehavioralCodes_CodeTimes'])), 
#                                             trial['AbsoluteTrialStartTime'])]

# # check
# print('bhv event time == bhv abs code time: '+ str((bhv_time==bhv_time2).all()))
# print('bhv event time == bhv code time + bhv abs time: ' + str((bhv_time==np.array(bhv_trial_time)+np.array(abs_time)).all()))

# # nwb - event_time
# n_bhv_code = nwbfile.processing['behavior']['MonkeyLogicEvents']['BehaviorMarkers'].data[:]
# n_bhv_time = nwbfile.processing['behavior']['MonkeyLogicEvents']['BehaviorMarkers'].timestamps[:]

# # nwb - code_time
# n_bhv_df = nwbfile.trials.to_dataframe()
# n_bhv_trial_time = [i for _, trial in n_bhv_df.iterrows() for i in json.loads(trial['BehavioralCodes_CodeTimes'])]

# # nwb - abs_code_time
# n_bhv_time2= [i+trial['AbsoluteTrialStartTime'] 
#              for _, trial in n_bhv_df.iterrows() for i in json.loads(trial['BehavioralCodes_CodeTimes'])]

# # nwb - abs_time
# n_abs_time = [i for _, trial in n_bhv_df.iterrows() 
#               for i in np.full(len(json.loads(trial['BehavioralCodes_CodeTimes'])), 
#                                               trial['AbsoluteTrialStartTime'])]

# # check
# print('nwb event time == nwb abs code time: '+ str(np.isclose(n_bhv_time, n_bhv_time2).all()))
# print('nwb event time == nwb code time + nwb abs time: ' + str(np.isclose(np.array(n_bhv_time), np.array(n_bhv_trial_time)+np.array(n_abs_time)).all()))
# print('nwb abs code time == nwb code time + nwb abs time: ' + str(np.isclose(n_bhv_time2, np.array(n_bhv_trial_time)+np.array(n_abs_time)).all()))


# # check
# time_difference = nwbfile.processing['behavior_ecephys_analysis']['TimeDifference']['time_difference'][0]

# print('bhv abs - nwb abs == time difference: ' + str((np.array(abs_time)-np.array(n_abs_time)==time_difference).all()))
# print('bhv event time - nwb event time == time_difference: ' + str((bhv_time-n_bhv_time==time_difference).all()))
# print('bhv code time == nwb code time: ' + str((bhv_trial_time==n_bhv_trial_time)))



#%% Extract bhv info
unitsdf = nwbfile.units.to_dataframe()
bhvdf = nwbfile.trials.to_dataframe()

# print(bhvdf)

bhv = bhvdf[['Trial', 'TrialError']].copy()
for col in ['targpos_to', 'targpos_go', 'targpos_tc', 'handpos_tc']:
    bhv[col] = np.nan
    bhv[col] = bhv[col].astype(object)

for i in bhv.index:
    status = np.array(json.loads(bhvdf.loc[i, 'ObjectStatusRecord_Status']))
    position = np.array(json.loads(bhvdf.loc[i, 'ObjectStatusRecord_Position']))

    to_targpos = position[(status[:, 0]==1)&(status[:, 4]==1)&(status.sum(axis=1)==2), 4]
    go_targpos = position[(status[:, 1]==1)&(status[:, 4]==1)&(status.sum(axis=1)==2), 4]
    touch_targpos = position[(status[:, 1]==1)&(status[:, 4]==1)&(status.sum(axis=1)==3), 4]

    object_col = 2 if bhvdf.loc[1, 'TrialError']==0 else 3
    touch_position = position[(status[:, 1]==1)&(status[:, 4]==1)&(status.sum(axis=1)==3), object_col] 
                              #[i for i in np.where(status.sum(axis=0)==1)[0] if status[np.where(status.sum(axis=1)==3)[0][0], i]==1][0]]

    
    bhv.at[i, 'targpos_to'] = to_targpos.squeeze().tolist() if len(to_targpos)>0 else [np.nan, np.nan] 
    bhv.at[i, 'targpos_go'] = go_targpos.squeeze().tolist() if len(go_targpos)>0 else [np.nan, np.nan] 
    bhv.at[i, 'targpos_tc'] = touch_targpos.squeeze().tolist() if len(touch_targpos)>0 else [np.nan, np.nan] 
    bhv.at[i, 'handpos_tc'] = touch_position.squeeze().tolist() if len(touch_position)>0 else [np.nan, np.nan] 

    codenum = json.loads(bhvdf.loc[i, 'BehavioralCodes_CodeNumbers'])
    codetime = json.loads(bhvdf.loc[i, 'BehavioralCodes_CodeTimes'])
    abs_start = bhvdf.loc[i, 'AbsoluteTrialStartTime']
    
    bhv.loc[i, 'to'] = codetime[codenum.index(3)] + abs_start if 3 in codenum else np.nan
    bhv.loc[i, 'go'] = codetime[codenum.index(4)] + abs_start if 4 in codenum else np.nan
    bhv.loc[i, 'mo'] = codetime[codenum.index(5)] + abs_start if 5 in codenum else np.nan
    bhv.loc[i, 'tc'] = codetime[codenum.index(6)] + abs_start if 6 in codenum else np.nan


#%% Check spike times (if cover the whole session)
# maxspktime = [max(i['spike_times']) for _, i in unitsdf.iterrows()]
# maxtrial = [np.argmax(i<=bhvdf['AbsoluteTrialStartTime'].values) for i in maxspktime]


#%% Pynapple interface
import pynapple as nap

nwb = nap.NWBFile(nwbfile)
units = nwb['units']
# units = units[units['sorter']=='TCR']
units = units[units['sorter']=='WaveClus']
corr_trials = bhv[bhv['TrialError']==0]

go_cue = nap.Tsd(t=corr_trials['go'].values,
                 d=corr_trials['targpos_go'].values, time_units='s')

mo_cue = nap.Tsd(t=corr_trials['mo'].values,
                 d=corr_trials['targpos_go'].values, time_units='s')

a7 = units[units['location']=='A7']
m1 = units[units['location']=='M1']
pmd = units[units['location']=='PMd']
s1 = units[units['location']=='S1']


peth0 = nap.compute_perievent(units[0], go_cue, minmax=(-0.2, 0.5), time_unit='s')
print(peth0)


#%% Plot peth and rasters
import matplotlib.pyplot as plt
import random

region = m1
for i in random.sample(list(region.index), 15):
    
    peth0 = nap.compute_perievent(region[i], mo_cue, minmax=(-0.2, 0.5), time_unit='s')

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(np.sum(peth0.count(0.01), 1), linewidth=3, color="red")
    plt.xlim(-0.2, 0.5)
    plt.ylabel("Count")
    plt.axvline(0.0)
    plt.subplot(212)
    plt.plot(peth0.to_tsd(), "|", markersize=1, color="red", mew=4)
    plt.xlabel("Time from stim (s)")
    plt.ylabel("Trial")
    plt.xlim(-0.2, 0.5)
    plt.axvline(0.0)
    plt.suptitle("PETH [GO-0.2, GO+0.5]")
    # plt.savefig(format('D://githubrepo//test//test.png'))
    plt.show()


#%%
print(1)