
def Inactivation(model, xt, timestart, timeend, localc):

  nTrial = xt.shape[0]
  assert nTrial == len(timestart)

  init_state_test1 = (torch.rand((1, model.hidden_size1))*0.3)
  init_state_test2 = (torch.rand((1, model.hidden_size2))*0.3)

  r1_np = np.zeros((nTrial, xt.shape[1], model.hidden_size1))
  r2_np = np.zeros((nTrial, xt.shape[1], model.hidden_size2))
  emg_np = np.zeros((nTrial, xt.shape[1], model.muscle_size))
  handv_np = np.zeros((nTrial, xt.shape[1], 2))

  model_ia = rRNN(model.input_size,
                  model.hidden_size1,
                  model.hidden_size2,
                  model.muscle_size)
  model_ia.load_state_dict(model.state_dict())
  if localc is not None:
    for localv in localc:
      if localv is 'w_ih1' or localv is 'w_ih2':
        nn.init.constant_(eval('model_ia.cell.'+localv+'.weight')[:, :2], 0)
      else:
        nn.init.constant_(eval('model_ia.cell.'+localv+'.weight'), 0)
      #nn.init.constant_(eval('model_ia.cell.'+localc+'.bias'), 0)

  for iTrial in range(nTrial):
    ## Epoch 1
    h1_test, r1_test, h2_test, r2_test, emg_test, handv_test = (
        model(xt[[iTrial], :timestart[iTrial], :],
              init_state_test1,
              init_state_test2)
    )

    ## Epoch 2
    # inactivation
    h1_test2, r1_test2, h2_test2, r2_test2, emg_test2, handv_test2 = (
        model_ia(xt[[iTrial], timestart[iTrial]:timeend[iTrial], :],
                 h1_test[[0], -1, :].detach(),
                 h2_test[[0], -1, :].detach())
    )

    ## Epoch 3
    # activation again
    h1_test3, r1_test3, h2_test3, r2_test3, emg_test3, handv_test3 = (
        model(xt[[iTrial], timeend[iTrial]:, :],
              h1_test2[[0], -1, :].detach(),
              h2_test2[[0], -1, :].detach())
    )

    r1_np[[iTrial], :, :] = (np.hstack([r1_test.detach().numpy(),
                                        r1_test2.detach().numpy(),
                                        r1_test3.detach().numpy()]))
    r2_np[[iTrial], :, :] = (np.hstack([r2_test.detach().numpy(),
                                        r2_test2.detach().numpy(),
                                        r2_test3.detach().numpy()]))
    emg_np[[iTrial], :, :] = (np.hstack([emg_test.detach().numpy(),
                                         emg_test2.detach().numpy(),
                                         emg_test3.detach().numpy()]))
    handv_np[[iTrial], :, :] = (np.hstack([handv_test.detach().numpy(),
                                           handv_test2.detach().numpy(),
                                           handv_test3.detach().numpy()]))

  pos_np = np.zeros_like(handv_np)
  for iTrial in range(nTrial):
      for iOutput in range(2):
          for iTime in np.arange(handv_np.shape[1]):
              pos_np[iTrial, iTime, iOutput] = integrate.trapz(
                  handv_np[iTrial, :iTime+1, iOutput], np.arange(iTime+1)/1000)

  # pos_np = np.zeros_like(handv_np)
  # for iTrial in range(pos_np.shape[0]):
  #   for iOutput in range(2):
  #       for iTime in np.arange(handv_test.detach().shape[1]):
  #           pos_np[iTrial, iTime, iOutput] = integrate.trapz(
  #               handv_np[iTrial, :iTime+1, iOutput], np.arange(iTime+1)/1000)
  #       for iTime in np.arange(handv_test2.detach().shape[1]):
  #           pos_np[iTrial, handv_test.detach().shape[1]+iTime, iOutput] = integrate.trapz(
  #               handv_np[iTrial,
  #                       handv_test.detach().shape[1]:handv_test.detach().shape[1]+iTime+1, iOutput],
  #                       np.arange(iTime+1)/1000)
  #       for iTime in np.arange(handv_test3.detach().shape[1]):
  #           pos_np[iTrial, handv_test.detach().shape[1]+iTime, iOutput] = integrate.trapz(
  #               handv_np[iTrial,
  #                       handv_test.detach().shape[1]:handv_test.detach().shape[1]+iTime+1, iOutput],
  #                       np.arange(iTime+1)/1000)

  result = {'r1': r1_np, 'r2': r2_np, 'emg': emg_np,
            'handv': handv_np, 'handpos': pos_np}

  return result, model_ia

#@title get validation-4 result (same endpoint)

SP = np.array([-240, -120, 0, 120, 240])/180*np.pi
nSP = len(SP)
nTest = nSP #200

# set specified time
time_delay_s =np.random.randint(200, size=nTest) + time_delay
time_reaction_s = 3*np.random.randn(nTest).astype(int) + time_reaction
time_move_s = 5*np.random.randn(nTest).astype(int) + time_move
#time_delay_s = time_delay_s.astype(int)
#time_reaction_s = time_reaction_s.astype(int)
#time_move_s = time_move_s.astype(int)

# generate inputs for specified time
if nTest == nSP:
  sp_array_test = SP[:, np.newaxis]
  ti_array_test = - sp_array_test*(time_delay_s + time_reaction_s + time_move_s)[:, np.newaxis]/1000
else:
  sp_array_test = SP[np.random.choice(range(len(SP)), (nTest, 1))]
  ti_array_test = (np.tile(np.arange(0, 360, 45)/180*np.pi, (1, int(nTest/8))).T
          - sp_array_test*(time_delay_s + time_reaction_s + time_move_s)[:, np.newaxis]/1000)


Cond_same_endpoint = Generate_flexible_timespecified(ti_array_test,
                                                     sp_array_test,
                                                     time_delay_s,
                                                     time_reaction_s,
                                                     time_move_s)
inputs_selected_test = Cond_same_endpoint.inputfunc(typename=typetype)
x_tensor_test = torch.from_numpy(np.array(inputs_selected_test, dtype=np.float32))
#print(x_tensor_test.shape)
Conditions_test = Cond_same_endpoint

now_rnn = rRNN(input_size, hidden_size1, hidden_size2, muscle_size)
now_rnn.load_state_dict(torch.load(load_path, map_location='cpu'))
result, now_rnn_ia = Inactivation(now_rnn, x_tensor_test[:, :, :],
                                  [Conditions_test.GO[i] for i in range(nTest)],
                                  [Conditions_test.MO[i] for i in range(nTest)],
                                  ['w_ih2'])

#@title get validation-4 result (random)

nTest = 50

model = now_rnn
SP = np.array([-240, -120, 0, 120, 240])/180*np.pi
sp_array_test = SP[np.random.choice(range(len(SP)), (nTest, 1))]
ti_array_test = np.random.rand(nTest, 1)*np.pi*2

Conditions_test = Generate_flexible_delayed(ti_array_test, sp_array_test)
inputs_selected_test = Conditions_test.inputfunc(typename=typetype)
x_tensor_test = torch.from_numpy(np.array(inputs_selected_test, dtype=np.float32))

now_rnn = rRNN(input_size, hidden_size1, hidden_size2, muscle_size)
now_rnn.load_state_dict(torch.load(load_path, map_location='cpu'))
result, now_rnn_ia = Inactivation(now_rnn, x_tensor_test[:, :, :],
                                  [Conditions_test.MO[i] for i in range(nTest)],
                                  [Conditions_test.MO[i]+100 for i in range(nTest)],
                                  ['w_h21'])

print(now_rnn.cell.w_ih2.bias.shape)