# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import task_generators.basic_setup as bs
import seaborn as sns


def plot_condition(input_type_name, time_dict_trial: dict, inputs, outputs, trial_info: dict):

    """

    Parameters
    ----------
    input_type_name
    time_dict_trial: should be a dict with keys 'time_points', 'go', 'mo'
    inputs: should be 2-D
    outputs: should be 2-D
    trial_info: should be a dict with keys 'i_trial' and 'SP'

    Returns
    -------

    """

    input_type_list = sum([bs.input_label_dict[k] for k in input_type_name], [])

    plt.figure()
    for idx, item in enumerate(input_type_list):
        ax1 = plt.subplot2grid((len(input_type_list) + 1, 2), (idx, 0), colspan=1, rowspan=1)
        # plt.subplot(len(input_type_list)+1, 2, (idx + 1) * 2 - 1)
        ax1.plot(time_dict_trial['time_points'], inputs[:time_dict_trial['time_total'], idx])
        plt.title('input-%d: %s' % (idx + 1, item))

    ax2 = plt.subplot2grid((len(input_type_list) + 1, 2), (len(input_type_list), 0), colspan=1, rowspan=1)
    # plt.subplot(len(input_type_list)+1, 2, len(input_type_list) * 2 + 1)
    ax2.plot(time_dict_trial['time_points'],
             np.zeros_like(time_dict_trial['time_points']), 'b')
    if time_dict_trial['go'] == time_dict_trial['mo']:
        ax2.plot([0, time_dict_trial['mo']], [0, 0], 'b', marker='.')
        ax2.set_xticks([0, time_dict_trial['mo'], 300])
        ax2.set_xticklabels(['TO', 'MO', '300'])
    else:
        ax2.plot([0, time_dict_trial['go'], time_dict_trial['mo']], [0, 0, 0], 'b', marker='.')
        ax2.set_xticks([0, time_dict_trial['go'], time_dict_trial['mo'], 300])
        ax2.set_xticklabels(['TO', 'GO', 'MO', '300'])
    plt.title('time markers')

    ax3 = plt.subplot2grid((len(input_type_list) + 1, 2), (0, 1), colspan=1, rowspan=2)
    # plt.subplot(2, 2, 2)
    ax3.plot(np.cos(np.linspace(0, np.pi * 2, 100)) * bs.R_CIRCLE,
             np.sin(np.linspace(0, np.pi * 2, 100)) * bs.R_CIRCLE,
             ls='--')
    ax3.plot(inputs[:time_dict_trial['time_total'], 0], inputs[:time_dict_trial['time_total'], 1], marker='.')
    ax3.plot(inputs[0, 0], inputs[0, 1], 'k', marker='>')
    ax3.plot(inputs[time_dict_trial['time_total']-1, 0], inputs[time_dict_trial['time_total']-1, 1], 'k', marker='s')
    plt.title('Trial %d ' % trial_info['i_trial'] + 'SP = {:.0f} d/s'.format(trial_info['SP'] / np.pi * 180))

    ax4 = plt.subplot2grid((len(input_type_list) + 1, 2), (2, 1), colspan=1, rowspan=1)
    # plt.subplot(2, 2, 4)
    ax4.plot(outputs[:time_dict_trial['time_total'], :])
    plt.title('output templates')

    ax5 = plt.subplot2grid((len(input_type_list) + 1, 2), (len(input_type_list), 1), colspan=1, rowspan=1)
    ax5.plot(time_dict_trial['time_points'],
             np.zeros_like(time_dict_trial['time_points']), 'b')
    if time_dict_trial['go'] == time_dict_trial['mo']:
        ax5.plot([0, time_dict_trial['mo']], [0, 0], 'b', marker='.')
        ax5.set_xticks([0, time_dict_trial['mo'], 300])
        ax5.set_xticklabels(['TO', 'MO', '300'])
    else:
        ax5.plot([0, time_dict_trial['go'], time_dict_trial['mo']], [0, 0, 0], 'b', marker='.')
        ax5.set_xticks([0, time_dict_trial['go'], time_dict_trial['mo'], 300])
        ax5.set_xticklabels(['TO', 'GO', 'MO', '300'])
    plt.title('time markers')

    plt.tight_layout()
    plt.show()


def plot_inputs_and_behavior(inputs, outputs, hand_v_np, trials_to_plots=None):
    n_test = hand_v_np.shape[0]

    if trials_to_plots is None:
        if n_test == 5:
            plot_idx = np.arange(n_test)
        else:
            plot_idx = np.random.choice(n_test, 5)
    else:
        plot_idx = trials_to_plots

    plt.figure(figsize=(20, 5))
    for i, iTrial in enumerate(plot_idx):
        plt.subplot(2, 5, i + 1)
        plt.plot(inputs[iTrial, :, :])

        plt.subplot(2, 5, i + 6)
        plt.plot(hand_v_np[iTrial, :, :])
        plt.plot(outputs[iTrial, :, :], '--')
    plt.subplots_adjust(wspace=0.3)
    plt.show()


def plot_hand_trajectory(inputs, conditions, hand_pos_np, trials_to_plots=None):
    n_test = hand_pos_np.shape[0]
    if trials_to_plots is None:
        if n_test == 5:
            plot_idx = np.arange(n_test)
        else:
            plot_idx = np.random.choice(n_test, 5)
    else:
        plot_idx = trials_to_plots

    pos_plot = hand_pos_np
    plt.figure(figsize=(20, 5))
    for i, iTrial in enumerate(plot_idx):

        plt.subplot(1, 5, i + 1)

        scatter_list = []
        cmap = plt.get_cmap('Greens', conditions.time_dict['time_total'][iTrial])
        cmap2 = plt.get_cmap('Blues', conditions.time_dict['time_total'][iTrial])

        for iTime in range(conditions.time_dict['time_total'][iTrial]):
            if (iTime + 1) % 10 == 0:
                plt.scatter(inputs[iTrial, iTime, 0],
                            inputs[iTrial, iTime, 1],
                            facecolors=cmap(iTime), edgecolors='none')

                plt.scatter(pos_plot[iTrial, iTime, 0],
                            pos_plot[iTrial, iTime, 1],
                            facecolors=cmap2(iTime), edgecolors='none',
                            s=8)

            if iTime == conditions.time_dict['time_total'][iTrial] - 1:
                scatter_a = plt.scatter(inputs[iTrial, iTime, 0],
                                        inputs[iTrial, iTime, 1],
                                        facecolors=cmap(iTime), edgecolors='none')

                scatter_list.append(scatter_a)
                scatter_i = plt.scatter(pos_plot[iTrial, iTime, 0],
                                        pos_plot[iTrial, iTime, 1],
                                        facecolors=cmap2(iTime), edgecolors='none',
                                        s=8)
                scatter_list.append(scatter_i)

        plt.scatter(inputs[iTrial, conditions.time_dict['time_total'][iTrial] - 1, 0],
                    inputs[iTrial, conditions.time_dict['time_total'][iTrial] - 1, 1],
                    s=50, c='g')
        plt.scatter(pos_plot[iTrial, conditions.time_dict['time_total'][iTrial] - 1, 0],
                    pos_plot[iTrial, conditions.time_dict['time_total'][iTrial] - 1, 1],
                    s=50, c='b')
        plt.plot(np.cos(np.linspace(0, np.pi * 2, 100)) * bs.R_CIRCLE,
                 np.sin(np.linspace(0, np.pi * 2, 100)) * bs.R_CIRCLE, ls='--', c='k')

        plt.axis('square')
        plt.title('Trial #%d ' % iTrial +
                  'SP = {:.0f} d/s'.format(conditions.condition_dict['target_speed'][iTrial][0] / np.pi * 180))
        plt.subplots_adjust(right=0.7, wspace=0.3)
        # plt.savefig(behavior_path + 'new_trial%d.png' % iTrial)

        if i == 5:
            plt.legend(scatter_list, ['target pos'],
                       bbox_to_anchor=(1.1, 1.2), loc="upper left", ncol=2)

    plt.show()


def plot_rate_heatmap(rates, title, trials_to_plots=None):
    n_test = rates.shape[0]
    if trials_to_plots is None:
        if n_test == 5:
            plot_idx = np.arange(n_test)
        else:
            plot_idx = np.random.choice(n_test, 5)
    else:
        plot_idx = trials_to_plots

    plt.figure(figsize=(20, 5))
    for i, iTrial in enumerate(plot_idx):
        plt.subplot(1, 5, i + 1)
        sns.heatmap(rates[iTrial, :, :].T)
    plt.subplots_adjust(wspace=0.3)
    plt.suptitle(title)
    plt.show()


def plot_3d_pca_trial_trajectory(pcs_plot, slice_start, slice_end, **kwargs):

    n_trial = pcs_plot.shape[0]
    if 'line_color_list' in kwargs.keys():
        line_color_list = kwargs['line_color_list']
    else:
        line_color_list = ['grey' for i in range(n_trial)]

    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(111, projection='3d')
    for i_trial in range(n_trial):
        x, y, z = (pcs_plot[i_trial, slice_start[i_trial]:slice_end[i_trial], 0],
                   pcs_plot[i_trial, slice_start[i_trial]:slice_end[i_trial], 1],
                   pcs_plot[i_trial, slice_start[i_trial]:slice_end[i_trial], 2])

        ax1.plot3D(x, y, z, color=line_color_list[i_trial])

        if 'scatter_list' in kwargs.keys():
            for scatter_type in kwargs['scatter_list']:
                scatter_time_list_i = scatter_type['time']
                scatter_color_list_i = scatter_type['color']
                scatter_marker_list_i = scatter_type['marker']

                # print(scatter_time_list_i)
                # print(scatter_color_list_i)
                ax1.scatter3D(x[scatter_time_list_i[i_trial]],
                              y[scatter_time_list_i[i_trial]],
                              z[scatter_time_list_i[i_trial]],
                              color=scatter_color_list_i[i_trial],
                              marker=scatter_marker_list_i)

    ax1.set_xticklabels('')
    ax1.set_yticklabels('')
    ax1.set_zticklabels('')
    ax1.grid(False)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.view_init(elev=45, azim=45)
    plt.show()


def set_colors(cond):
    n_trial = len(cond.time_dict['time_total'])
    speed_colors = [plt.cm.tab10(speed_idx)
                    for speed_idx in range(len(np.unique(cond.condition_dict['target_speed'])))]
    _, _, color_idx = np.unique(cond.condition_dict['target_speed'], return_index=True, return_inverse=True)
    speed_color_idx = [speed_colors[i] for i in color_idx]

    angle_colors = [plt.cm.hsv(angle) for angle in range(360)]
    target_pos_to = [cond.condition_dict['target_init'][i][0] for i in range(n_trial)]
    target_pos_to = np.array(target_pos_to)/np.pi*180
    angle_color_idx_to = [angle_colors[i] for i in target_pos_to.astype(int)]

    target_pos_go = [cond.object_dict['target_rad'][i][cond.time_dict['go'][i]-1][0] for i in range(n_trial)]
    target_pos_go = np.array(target_pos_go)/np.pi*180
    angle_color_idx_go = [angle_colors[i] for i in target_pos_go.astype(int)]

    target_pos_mo = [cond.object_dict['target_rad'][i][cond.time_dict['mo'][i]-1][0] for i in range(n_trial)]
    target_pos_mo = np.array(target_pos_mo)/np.pi*180
    angle_color_idx_mo = [angle_colors[i] for i in target_pos_mo.astype(int)]

    target_pos_end = [cond.object_dict['target_rad'][i][cond.time_dict['time_total'][i]-1][0]
                      for i in range(n_trial)]
    target_pos_end = np.array(target_pos_end)/np.pi*180
    angle_color_idx_end = [angle_colors[i] for i in target_pos_end.astype(int)]

    return speed_color_idx, {'to': angle_color_idx_to, 'go': angle_color_idx_go,
                             'mo': angle_color_idx_mo, 'end': angle_color_idx_end}
