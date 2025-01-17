# -*- coding: utf-8 -*-
"""

To generate input and output template for no-delay reaching (center-out) tasks.
"""

import numpy as np
from sklearn.decomposition import PCA


def get_time_slice(rate_np, slice_start, slice_end):
    assert rate_np.shape[0] == len(slice_start)
    assert rate_np.shape[0] == len(slice_end)

    (n_trial, n_time, n_neuron) = rate_np.shape
    max_slice_bin = max(slice_end - slice_start)
    rate_slice = np.zeros((n_trial, max_slice_bin, n_neuron))
    for i_trial in range(n_trial):
        slice_i = rate_np[i_trial, slice_start[i_trial]:slice_end[i_trial], :]
        rate_slice[i_trial, :slice_i.shape[0], :] = slice_i

    return rate_slice


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


def cal_diff(f, h, order, accuracy):
    if accuracy == 2:
        if order == 1:
            diff1 = [0]
            for i in np.arange(1, len(f)-1):
                diff1_i = (f[i+1]-f[i-1])/(2*h)
                diff1.append(diff1_i)
            diff1.append(0)
            return np.array(diff1)
        elif order == 2:
            diff2 = [0]
            for i in np.arange(1, len(f)-1):
                diff2_i = (f[i+1]-2*f[i]+f[i-1])/h**2
                diff2.append(diff2_i)
            diff2.append(0)
            return np.array(diff2)
        elif order == 3:
            diff3 = [0, 0]
            for i in np.arange(2, len(f)-2):
                diff3_i = (f[i+2]-f[i+1]+2*f[i-1]-f[i-2])/(2*h**3)
                diff3.append(diff3_i)
            diff3 = diff3 + [0, 0]
            return np.array(diff3)
    elif accuracy == 4:
        if order == 1:
            diff1 = [0, 0]
            for i in np.arange(2, len(f)-2):
                diff1_i = (-f[i+2]+8*f[i+1]-8*f[i-1]+f[i-2])/(12*h)
                diff1.append(diff1_i)
            diff1 = diff1 + [0, 0]
            return np.array(diff1)
        elif order == 2:
            diff2 = [0, 0]
            for i in np.arange(2, len(f)-2):
                diff2_i = (-f[i+2]+16*f[i+1]-30*f[0]+16*f[i-1]-f[i-2])/(12*h**2)
                diff2.append(diff2_i)
            diff2 = diff2 + [0, 0]
            return np.array(diff2)
        elif order == 3:
            diff3 = [0, 0, 0]
            for i in np.arange(3, len(f)-3):
                diff3_i = (-f[i+3]+8*f[i+2]-13*f[i+1]+13*f[i-1]-8*f[i-2]+f[i-3])/(8*h**3)
                diff3.append(diff3_i)
            diff3 = diff3 + [0, 0, 0]
            return np.array(diff3)


def cal_curvature(arr, h):
    # first derivative
    x0, y0, z0 = arr[:, 0], arr[:, 1], arr[:, 2]
    # x1, y1, z1 = np.diff(x0), np.diff(y0), np.diff(z0)
    # x2, y2, z2 = np.diff(x1), np.diff(y1), np.diff(z1)
    x1 = cal_diff(x0, h, 1, 2)
    y1 = cal_diff(y0, h, 1, 2)
    z1 = cal_diff(z0, h, 1, 2)

    x2 = cal_diff(x0, h, 2, 4)
    y2 = cal_diff(y0, h, 2, 4)
    z2 = cal_diff(z0, h, 2, 4)

    # |c1xc2|/|c1|^3
    k = (np.sqrt((y1*z2-y2*z1)**2+(z1*x2-x1*z2)**2+(x1*y2-x2*y1)**2) / (np.sqrt(x1**2+y1**2+z1**2))**3)

    return k


def cal_torsion(arr, h):
    # first derivative
    x0, y0, z0 = arr[:, 0], arr[:, 1], arr[:, 2]
    # x1, y1, z1 = np.diff(x0), np.diff(y0), np.diff(z0)
    # x2, y2, z2 = np.diff(x1), np.diff(y1), np.diff(z1)
    x1 = cal_diff(x0, h, 1, 2)
    y1 = cal_diff(y0, h, 1, 2)
    z1 = cal_diff(z0, h, 1, 2)

    x2 = cal_diff(x0, h, 2, 4)
    y2 = cal_diff(y0, h, 2, 4)
    z2 = cal_diff(z0, h, 2, 4)

    x3 = cal_diff(x0, h, 3, 4)
    y3 = cal_diff(y0, h, 3, 4)
    z3 = cal_diff(z0, h, 3, 4)

    temp1 = np.array(([[x1, x2, x3], [y1, y2, y3], [z1, z2, z3]]))
    temp2 = (y1*z2-y2*z1)**2+(z1*x2-x1*z2)**2+(x1*y2-x2*y1)**2

    # det(c1, c2, c3)/|c1xc2|^2
    tau = []
    for i in range(arr.shape[0]):
        tau_i = np.linalg.det(temp1[:, :, i]) /temp2[i]
        tau.append(tau_i)

    return np.array(tau)


def get_trajectory_length(traj, time_idx):
    n_trial = traj.shape[0]
    n_time = traj.shape[1]

    step_len_traj = np.zeros((n_trial, n_time))
    cum_len_traj = np.zeros((n_trial, n_time))
    len_traj = np.zeros((n_trial,))
    for i_trial in range(n_trial):
        for i_time in np.arange(1, time_idx[i_trial]):
            step_len_traj[i_trial, i_time] = np.linalg.norm(
                traj[i_trial, i_time, :] - traj[i_trial, i_time-1, :])
            cum_len_traj[i_trial, i_time] = np.sum(step_len_traj[i_trial, :i_time+1])
        len_traj[i_trial] = np.sum(step_len_traj[i_trial, :])
        assert (len_traj[i_trial] - max(cum_len_traj[i_trial, :])) < 1e-10
        step_len_traj[i_trial, :] /= max(step_len_traj[i_trial, :])
        # cum_len_traj[i_trial, :] /= max(cum_len_traj[i_trial, :])

    return step_len_traj, cum_len_traj, len_traj


def get_trajectory_curvature(traj, interval):
    # the time bin of traj should be this interval
    n_trial = traj.shape[0]
    n_time = traj.shape[1]

    curvature_traj = np.zeros((n_trial, n_time))
    for i_trial in range(n_trial):
        kk = cal_curvature(traj[i_trial, :, :], interval / 1000)
        curvature_traj[i_trial, :len(kk)] = kk
    curvature_traj[np.isnan(curvature_traj)] = 0

    normed_ct = curvature_traj.copy()
    for i_trial in range(n_trial):
        normed_ct[i_trial, :] /= max(normed_ct[i_trial, :])

    return curvature_traj, normed_ct


def get_trajectory_torsion(traj, interval):
    # the time bin of traj should be this interval
    n_trial = traj.shape[0]
    n_time = traj.shape[1]

    torsion_traj = np.zeros((n_trial, n_time))
    for i_trial in range(n_trial):
        tt = cal_torsion(traj[i_trial, :, :], interval / 1000)
        torsion_traj[i_trial, :len(tt)] = tt
    torsion_traj[np.isnan(torsion_traj)] = 0

    normed_tt = torsion_traj.copy()
    for i_trial in range(n_trial):
        normed_tt[i_trial, :] = ((torsion_traj[i_trial, :] - np.mean(torsion_traj[i_trial, :])) /
                                    (max(torsion_traj[i_trial, :]) - min(torsion_traj[i_trial, :])))

    return torsion_traj, normed_tt
