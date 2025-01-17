# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import task_generators.delayed_tasks as dd
import task_generators.no_delay_tasks as nd
from scipy import integrate
from network_settings.muscleRNN_settings import MuscleRNN
import task_generators.basic_setup as bs


def validation(model, num_test, device, sp_dict, task_type, **kwargs):

    generate = None
    if task_type[0] == 'no-delay':
        generate = nd.NoDelayGenerator
    elif task_type[0] == 'delayed':
        generate = dd.StandardCondition
    elif task_type[0] == 'time_specified':
        generate = dd.TimeSpecifiedCondition
    elif task_type[0] == 'position_specified':
        generate = dd.StandardCondition
    elif task_type[0] == 'memory':
        generate = dd.TargetDisappearCondition
    elif task_type[0] == 'cued_memory':
        generate = dd.CuedTargetDisappearCondition

    target_speeds = sp_dict[task_type[1]]
    sp_array_test = target_speeds[np.random.choice(range(len(target_speeds)), (num_test, 1))]
    ti_array_test = np.random.rand(num_test, 1)*np.pi*2

    if task_type[0] == 'time_specified':
        specified_dict = kwargs['specified_dict']
        if 'init_position' in specified_dict.keys():
            ti_array_test = (specified_dict['init_position'] -
                             sp_array_test * (specified_dict['time_delay'] +
                                              specified_dict['time_reaction'] +
                                              specified_dict['time_move'])[:, np.newaxis] / 1000)
        else:
            ti_array_test = - sp_array_test * (specified_dict['time_delay'] +
                                               specified_dict['time_reaction'] +
                                               specified_dict['time_move'])[:, np.newaxis] / 1000

    if task_type[0] == 'position_specified':
        specified_dict = kwargs['specified_dict']
        ti_array_test = specified_dict['init_position']
        conditions_test = generate(ti_array_test, sp_array_test)
    else:
        conditions_test = generate(ti_array_test, sp_array_test, **kwargs)

    inputs_selected_test = conditions_test.inputs('a1')
    outputs_selected_test = conditions_test.outputs()

    x_tensor_test = torch.from_numpy(np.array(inputs_selected_test, dtype=np.float32))
    x_tensor_test = x_tensor_test.to(device)

    init_state_test1 = (torch.zeros((1, model.hidden_size1))).to(device)
    init_state_test2 = (torch.zeros((1, model.hidden_size2))).to(device)
    # init_state_test1 = (torch.rand((1, model.hidden_size1))*0.3).to(device)
    # init_state_test2 = (torch.rand((1, model.hidden_size2))*0.3).to(device)
    h1_test, r1_test, h2_test, r2_test, emg_test, hand_v_test = model(x_tensor_test,
                                                                      init_state_test1,
                                                                      init_state_test2)

    r1_np = r1_test.cpu().detach().numpy()
    r2_np = r2_test.cpu().detach().numpy()
    emg_np = emg_test.cpu().detach().numpy()
    hand_v_np = hand_v_test.cpu().detach().numpy()

    pos_np = np.zeros_like(hand_v_np)
    for i_trial in range(num_test):
        for i_output in range(2):
            for i_time in np.arange(hand_v_np.shape[1]):
                temp = hand_v_np.copy()
                # print(temp.shape)
                temp[i_trial, :conditions_test.time_dict['mo'][i_trial], :] = 0
                pos_np[i_trial, i_time, i_output] = integrate.trapz(
                        temp[i_trial, :(i_time+1), i_output],
                        np.arange(i_time+1)/1000)

    result = {'r1': r1_np, 'r2': r2_np, 'emg': emg_np, 'hand_v': hand_v_np,
              'hand_pos': pos_np}

    return conditions_test, result, inputs_selected_test, outputs_selected_test


if __name__ == '__main__':

    n_test = 5

    input_size = 3
    hidden_size1 = 200
    hidden_size2 = 200
    muscle_size = 6

    type_name = 'a1'
    device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    load_path = 'F://Aac//Work//Project//newIC//results//m4_nmp_delayed_inter5sp_0115.pth'

    t_rnn = MuscleRNN(input_size, hidden_size1, hidden_size2, muscle_size).to(device0)
    t_rnn.load_state_dict(torch.load(load_path))

    c1, r1, i1, o1 = validation(t_rnn, n_test, device0, bs.sp_dict,
                                ['no-delay', 'reaching'])

    # print(r1['hand_v'].shape)
    plt.figure()
    plt.plot(r1['hand_v'][0, :, :])
    plt.plot(o1[0, :, :], ls='--')
    plt.show()

    c2, r2, i2, o2 = validation(t_rnn, n_test, device0, bs.sp_dict,
                                ['delayed', 'inter-5sp'])

    # print(r1['hand_v'].shape)
    plt.figure()
    plt.plot(r2['hand_v'][0, :, :])
    plt.plot(o2[0, :, :], ls='--')
    plt.show()

    # set specified time
    time_delay_s = np.random.randint(200, size=n_test) + bs.time_delay
    time_reaction_s = 3*np.random.randn(n_test).astype(int) + bs.time_reaction
    time_move_s = 5*np.random.randn(n_test).astype(int) + bs.time_move

    specified_dict = {'time_delay': time_delay_s,
                      'time_reaction': time_reaction_s,
                      'time_move': time_move_s}

    c3, r3, i3, o3 = validation(t_rnn, n_test, device0, bs.sp_dict,
                                ['time_specified', 'inter-5sp'],
                                specified_dict=specified_dict)

    # print(r1['hand_v'].shape)
    plt.figure()
    plt.plot(r3['hand_v'][0, :, :])
    plt.plot(o3[0, :, :], ls='--')
    plt.show()

    c4, r4, i4, o4 = validation(t_rnn, n_test, device0, bs.sp_dict,
                                ['memory', 'inter-5sp'],
                                tau_dict={'ref_marker': 'go', 'duration': [0, 'inf']})

    plt.figure()
    plt.plot(r4['hand_v'][0, :, :])
    plt.plot(o4[0, :, :], ls='--')
    plt.show()

    c5, r5, i5, o5 = validation(t_rnn, n_test, device0, bs.sp_dict,
                                ['cued_memory', 'inter-5sp'],
                                tau_dict={'ref_marker': 'go', 'duration': [-50, 50]}, d_ratio=0.5)

    plt.figure()
    plt.plot(r5['hand_v'][0, :, :])
    plt.plot(o5[0, :, :], ls='--')
    plt.show()
