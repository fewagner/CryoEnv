import numpy as np
from cryoenv.envs._discretization import action_to_discrete, action_from_discrete, observation_from_discrete, \
    observation_to_discrete

V_SET_IV = (0, 99)
V_SET_STEP = 1
PH_IV = (0, 0.999)
PH_STEP = 0.001
WAIT_IV = (2, 100)
WAIT_STEP = 2
NMBR_CHANNELS = 1

# choose action
reset = np.array([False])
dV = np.array([14])
wait = np.array(10)
do_action = False

# choose observation
V = [np.array([10]),]
ph = [np.array([0.001]),]
nmbr_channels = 1
do_observation = True

# --------------------------------------------------
# dont change anything below this line
# --------------------------------------------------
if do_action:
    print('\nACTION TO DISCRETE')
    discrete = action_to_discrete(reset, dV, wait,
                                  wait_iv=WAIT_IV, V_iv=V_SET_IV, wait_step=WAIT_STEP,
                                  V_step=V_SET_STEP)

    print('\nACTION FROM DISCRETE')
    from_discrete = action_from_discrete(discrete, nmbr_channels=NMBR_CHANNELS,
                                         wait_iv=WAIT_IV, V_iv=V_SET_IV, wait_step=WAIT_STEP,
                                         V_step=V_SET_STEP)

    print('\nSUMMARY')
    print('origin: ', reset, dV, wait)
    print('conv: ', from_discrete)

if do_observation:
    for v, p in zip(V, ph):
        print('\nOBSERVATION TO DISCRETE')
        discrete = observation_to_discrete(v, p, V_iv=V_SET_IV, ph_iv=PH_IV, V_step=V_SET_STEP,
                                           ph_step=PH_STEP)

        print('\nOBSERVATION FROM DISCRETE')
        from_discrete = observation_from_discrete(discrete, nmbr_channels=NMBR_CHANNELS, V_iv=V_SET_IV,
                                                  ph_iv=PH_IV,
                                                  V_step=V_SET_STEP, ph_step=PH_STEP)

        print('\nSUMMARY')
        print('origin: ', v, p)
        print('conv: ', from_discrete)
