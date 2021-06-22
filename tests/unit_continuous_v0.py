import gym
import numpy as np

import warnings

warnings.simplefilter('ignore')
np.random.seed(0)
gym.logger.set_level(40)

if __name__ == '__main__':
    # ------------------------------------------------
    # CREATE THE ENVIRONMENT
    # ------------------------------------------------

    # constants
    env_kwargs = {
        'V_set_iv': (0, 99),
        'V_set_step': 1,
        'ph_iv': (0, 0.99),
        'ph_step': 0.01,
        'wait_iv': (10, 50),
        'wait_step': 2,
        'heater_resistance': np.array([100.]),
        'thermal_link_channels': np.array([[1.]]),
        'thermal_link_heatbath': np.array([1.]),
        'temperature_heatbath': 0.,
        'min_ph': 0.1,
        'g': np.array([0.0001]),
        'T_hyst': np.array([0.1]),
        'T_hyst_reset': np.array([0.9]),
        'hyst_wait': np.array([50]),
        'control_pulse_amplitude': 50,
        'env_fluctuations': 0.005,
        'model_pileup_drops': True,
        'prob_drop': np.array([1e-3]),  # per second!
        'prob_pileup': np.array([0.1]),
        'save_trajectory': True,
        'k': np.array([15]),
        'T0': np.array([0.5]),
    }

    print('Create Environment.')
    env = gym.make('cryoenv:cryoenv-continuous-v0',
                   **env_kwargs,
                   )

    trajectory = [[[40, 10], 'here we should see pulses'],
                  [[40, 10], ''],
                  [[0, 10], 'here the saturation should kick'],
                  [[0, 10], ''],
                  [[40, 10], 'here we should see no more pulses'],
                  [[40, 10], ''],
                  [[99, 50], 'here the saturation reset'],
                  [[99, 50], ''],
                  [[40, 10], 'here we should see pulses'],
                  [[40, 10], ''],
                  ]

    for a, m in trajectory:
        new_state, reward, _, _ = env._step(np.array(a))
        if len(m) > 0:
            print(m)
        print(f'action: {a}, new_state: {new_state}, reward: {reward}')
