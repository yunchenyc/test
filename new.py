#!/usr/bin/env python
# coding: utf-8
'''
 @Author: yunchen
 @Time: 2025/1/17

'''

#%% Import
from pynwb import NWBHDF5IO
import os
import json
import numpy as np
import quantities as pq
from scipy.stats import sem
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import seaborn as sns

#%% Load nwb file
dirpath = format('F://Aac//Work//Project//Interception//Data//abel_20240529')

# # integrated data
filepath = os.path.join(dirpath, 'standard_data.nwb')
io = NWBHDF5IO(filepath, mode='r')
nwbfile = io.read()
print(nwbfile)


#%% Extract bhv info
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

bhv['TrialErorr'] = bhvdf['TrialError']
bhv['TargetVelocity'] = bhvdf['UserVars_angularV']
bhv['TrialError_annotation'] = bhvdf['TrialError_annotation']

del bhvdf


#%% Pynapple interface
import pynapple as nap

nwb = nap.NWBFile(nwbfile)
del nwbfile

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

def get_spike_times(spiketimes, event_marker, time_window:tuple, time_unit='s'):
    '''
    Function to get spike times around event marker
    Args:
        spiketimes: TsGroup of spike times
        event_marker: Tsd of event markers
        time_window: tuple of time window, as (min, max)
        time_unit: time unit, default as second
    Returns:
        peth: dict of TsGrjoups
        peth_array: dict of lists of peri-event time histogram, 
                    e.g. peth[unit_index] is a list of spike times for each trial around the event marker
    '''
    
    peth = nap.compute_perievent(spiketimes, event_marker, minmax=time_window, time_unit=time_unit)
    
    if time_unit == 's':
        pq_unit = pq.s
    elif time_unit == 'ms':
        pq_unit = pq.ms
    else:
        raise ValueError("Unsupported time unit. Use 's' for seconds or 'ms' for milliseconds.")
    
    peth_array = {k: [np.array(t.index)*pq_unit for _, t in v.items()] for k, v in peth.items()}
    
    return peth, peth_array


def get_spike_counts(spiketimes, event_marker, time_window:tuple, binsize=0.01, time_unit='s', convert=False):
    '''
    Function to get spike counts around event marker
    Args:
        spiketimes: TsGroup of spike times
        event_marker: Tsd of event markers
        time_window: tuple of time window, as (min, max)
        binsize: bin size for spike count
        time_unit: time unit, default as second
    Returns:    
        peth_counts: dict of spike counts
        peth_array_ktn: numpy array of peri-event time histogram, (n_trial, n_bin, n_unit)
    '''

    peth = nap.compute_perievent(spiketimes, event_marker, minmax=time_window, time_unit=time_unit)
    
    peth_counts = {k: v.count(binsize) for k, v in peth.items()}
    peth_array = np.array([v for _, v in peth_counts.items()])
    peth_array_ktn = np.swapaxes(peth_array, 0, 2)

    return peth_counts, peth_array_ktn


def get_spike_rates(spiketimes, event_marker, time_window:tuple, binsize=0.01, smooth_std=0.04, time_unit='s'):
    '''
    Function to get spike rates around event marker
    Args:       
        spiketimes: TsGroup of spike times
        event_marker: Tsd of event markers
        time_window: tuple of time window, as (min, max)
        binsize: bin size for spike count
        smooth_std: standard deviation for Gaussian smoothing
        time_unit: time unit, default as second
    Returns:
        peth_rate_smooth_array_ktn: numpy array of smoothed peri-event time histogram, (n_trial, n_bin, n_unit)
    '''
    peth = nap.compute_perievent(spiketimes, event_marker, minmax=time_window, time_unit=time_unit)
    peth_rate = {k: v.count(binsize)/binsize for k, v in peth.items()}
   
    if smooth_std == 0:
        peth_rate_array = np.array([v for _, v in peth_rate.items()])
        peth_rate_array_ktn = np.swapaxes(peth_rate_array, 0, 2)
        return peth_rate, peth_rate_array_ktn
   
    elif smooth_std > 0:
        peth_rate_smooth = {k: v.smooth(std=smooth_std) for k, v in peth_rate.items()}
        peth_rate_smooth_array = np.array([v for _, v in peth_rate_smooth.items()])
        peth_rate_smooth_array_ktn = np.swapaxes(peth_rate_smooth_array, 0, 2)
        return peth_rate_smooth, peth_rate_smooth_array_ktn
    else:
        raise ValueError("Unsupported smmoth_std value. Use 0 for no smoothing or a positive value for Gaussian smoothing.")


def get_mean_sem(arr, axis=0):
    if isinstance(arr, np.ndarray):
        arr_mean = np.mean(arr, axis=axis)
        arr_sem = sem(arr, axis=axis)
    elif isinstance(arr, nap.core.time_series.TsdFrame):
        arr_mean = arr.mean(axis=axis)
        arr_sem = sem(arr, axis=axis)
    return arr_mean, arr_sem


#%%
align_marker = mo_cue
window = (-0.2, 0.2)
# peth_go = nap.compute_perievent(units, go_cue, minmax=(-0.2, 0.5), time_unit='s')

st, st_arr = get_spike_times(units, align_marker, window)
sc, sc_arr = get_spike_counts(units, align_marker, window)
fr, fr_arr = get_spike_rates(units, align_marker, window)



#%% Plot peth and rasters

region = m1
for i in random.sample(list(region.index), 15):
    mean_fr, sem_fr = get_mean_sem(fr[i], axis=1) # trial-averaged
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(mean_fr, linewidth=3, color="red")
    # plt.xlim(-0.2, 0.5)
    plt.ylabel("Count")
    plt.axvline(0.0)
    plt.subplot(212)
    # peth = nap.compute_perievent(units[i], go_cue, minmax=(-0.2, 0.5))
    # plt.plot(peth.to_tsd(), "|", markersize=1, color="red", mew=4)
    plt.plot(st[i].to_tsd(), "|", markersize=1, color="red", mew=4)
    plt.xlabel("Time from stim (s)")
    plt.ylabel("Trial")
    # plt.xlim(-0.2, 0.5)
    plt.axvline(0.0)
    plt.suptitle("PETH [GO-0.2, GO+0.5]")
    # plt.savefig(format('D://githubrepo//test//test.png'))
    plt.show()

#%%
all_mean_sc = np.array([get_mean_sem(sc[i], axis=1)[0] for i in range(len(sc))])


region = m1

plt.figure()
plt.imshow(all_mean_sc[region.index, :], aspect='auto', cmap='copper')
plt.colorbar()
plt.show()

#%%
region = m1
for i in random.sample(list(region.index), 15):
    plt.figure()
    plt.imshow(sc_arr[:, :, 1], aspect='auto', cmap='copper')
    plt.show()

#%%
all_mean_fr = np.array([get_mean_sem(fr[i], axis=1)[0] for i in range(len(fr))])
time_axis = fr[0].index

all_m1 = all_mean_fr[m1.index, :]
all_s1 = all_mean_fr[s1.index, :]
all_pmd = all_mean_fr[pmd.index, :]
all_a7 = all_mean_fr[a7.index, :]

plt.figure()
plt.plot(time_axis, all_m1.mean(axis=0), label='M1')
plt.plot(time_axis, all_s1.mean(axis=0), label='S1')
plt.plot(time_axis, all_pmd.mean(axis=0), label='PMd')
plt.plot(time_axis, all_a7.mean(axis=0), label='A7')
plt.vlines(x=0, ymin=0, ymax=15, color='black', linestyle='--')
plt.legend()
plt.show()


#%% 
print('N(M1) = ', len(m1))
print('N(S1) = ', len(s1))
print('N(PMd) = ', len(pmd))
print('N(A7) = ', len(a7))


#%%
def get_trial_pca_trajectory(rate_np, n_pc):
    (n_trial, n_time, n_neuron) = rate_np.shape
    k_nt_n = rate_np.copy()
    k_nt_n = k_nt_n / (k_nt_n.max() + 5)

    k_nt = np.zeros((n_neuron, n_trial * n_time))
    for i_neuron in range(n_neuron):
        k_nt[i_neuron, :] = k_nt_n[:, :, i_neuron].reshape(n_trial * n_time, )

    pca = PCA(n_components=n_pc)
    pcs = pca.fit_transform(k_nt.T)
    evr = pca.explained_variance_ratio_

    pcs_plot = np.zeros((n_trial, n_time, n_pc))
    for i_pc in range(n_pc):
        pcs_plot[:, :, i_pc] = pcs[:, i_pc].reshape(n_trial, n_time)

    return pcs_plot, evr

m1_pcs_plot, m1_evr = get_trial_pca_trajectory(fr_arr[:, :, m1.index], 3)
s1_pcs_plot, s1_evr = get_trial_pca_trajectory(fr_arr[:, :, s1.index], 3)
pmd_pcs_plot, pmd_evr = get_trial_pca_trajectory(fr_arr[:, :, pmd.index], 3)
a7_pcs_plot, a7_evr = get_trial_pca_trajectory(fr_arr[:, :, a7.index], 3)

print('PCs Explained Variance Ratio (M1): ', m1_evr)
print('PCs Explained Variance Ratio (S1): ', s1_evr)
print('PCs Explained Variance Ratio (PMd): ', pmd_evr)
print('PCs Explained Variance Ratio (A7): ', a7_evr)

#%% General time distribution

delay_time = bhv['go'] - bhv['to']
reaction_time = bhv['mo'] - bhv['go']
movement_time = bhv['tc'] - bhv['mo']

plt.figure(dpi=300)
plt.subplot(311)
plt.hist(delay_time, bins=20, facecolor='white', edgecolor='black')
plt.xlabel('Delay Time (s)')
plt.ylabel('Count')

plt.subplot(312)
plt.hist(reaction_time, bins=20, facecolor='white', edgecolor='blue')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Count')

plt.subplot(313)
plt.hist(movement_time, bins=20, facecolor='white', edgecolor='red')
plt.xlabel('Movement Time (s)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


#%%
tv, tv_index, tv_count = np.unique(bhv['TargetVelocity'], return_inverse=True, return_counts=True)
tv_label = ['%s %d \u00B0/s' % ('CCW' if v > 0 else 'CW' if v < 0 else '', abs(v/np.pi*180)) for v in tv]

plt.figure()
plt.pie(tv_count, labels=tv_label, autopct='%1.1f%%')
plt.show()

bhv['delay_time'] = delay_time
bhv['reaction_time'] = reaction_time
bhv['movement_time'] = movement_time
bhv['tv_index'] = tv_index

plt.figure(dpi=300)
sns.boxplot(data=bhv, x='delay_time', hue='tv_index', dodge=True)
plt.xlabel('Delay Time (s)')
plt.ylabel('Target Velocity')
plt.legend(title='Target Velocity', loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Histogram of Delay Time Grouped by Target Velocity')
plt.show()

plt.figure(dpi=300)
sns.boxplot(data=bhv, x='reaction_time', hue='tv_index', dodge=True)
plt.xlabel('Reaction Time (s)')
plt.ylabel('Target Velocity')
plt.legend(title='Target Velocity', loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Histogram of Reaction Time Grouped by Target Velocity')
plt.show()

plt.figure(dpi=300)
sns.boxplot(data=bhv, x='movement_time', hue='tv_index', dodge=True)
plt.xlabel('Movement Time (s)')
plt.ylabel('Target Velocity')
plt.legend(title='Target Velocity', loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Histogram of Movement Time Grouped by Target Velocity')
plt.show()

#%%
te, te_index, te_count = np.unique(bhv['TrialError'], return_inverse=True, return_counts=True)
te_label = np.unique(bhvdf['TrialError_annotation'])

plt.figure()
plt.pie(te_count, labels=te, autopct='%1.1f%%')
plt.legend([f'{t}: {l}' for t, l in zip(te, te_label)], title='Trial Error', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


#%% 
from scipy import stats

corr_delay_time = delay_time[bhv['TrialError']==0]
corr_reaction_time = reaction_time[bhv['TrialError']==0]
corr_movement_time = movement_time[bhv['TrialError']==0]

# normal distribution test
ks_test = stats.kstest(corr_delay_time, 'norm')
print('K-S test for delay time, p = %f' % ks_test.pvalue)
print('delay time %s follow normal distribution' % ('does not' if ks_test.pvalue<0.05 else 'does'))

ks_test = stats.kstest(corr_reaction_time, 'norm')
print('K-S test for delay time, p = %f' % ks_test.pvalue)
print('delay time %s follow normal distribution' % ('does not' if ks_test.pvalue<0.05 else 'does'))

ks_test = stats.kstest(corr_movement_time, 'norm')
print('K-S test for delay time, p = %f' % ks_test.pvalue)
print('delay time %s follow normal distribution' % ('does not' if ks_test.pvalue<0.05 else 'does'))
