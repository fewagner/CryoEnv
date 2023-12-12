#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import matplotlib.pyplot as plt
import os
import json
import gym
import warnings

import sys
import time
# from IPython.display import display, clear_output
import torch
import os
import argparse

from cryoenv.mqtt import SoftActorCritic, ReturnTracker, ReplayBuffer, check, subscribe, publish, connect_mqtt, generate_sweep
from cryoenv.envs import CryoEnvSigWrapper

from mqtt_protocol import *
from on_message import receive_as_control


parser = argparse.ArgumentParser(description='Control CCS with a SoftActorCritic.')
parser.add_argument('-r', '--inference', action='store_true', help='activate for inference, dont for training')
args = vars(parser.parse_args())
print('Inputs processed.')

# In[2]:


np.random.seed(rseed)


# In[3]:


warnings.simplefilter('ignore')


# In[4]:


client_id = 'control-secondary'


# In[5]:


env = gym.make('cryoenv:cryoenv-sig-v0',
                   omega=omega,
                   sample_pars=False,
                   tpa_in_state=tpa_in_state,
                   pars={'store_raw': False,
                         'max_buffer_len': buffer_size,
                         'tpa_queue': tpa_queue,
                         'pileup_prob': pileup_prob,
                         'xi': np.array([xi]),
                         'tau': np.array([tau]),},
               render_mode='human',
                   )
print('Env made.')
print('Env observation space: ', env.observation_space.shape)
print('Env action space: ', env.action_space.shape)

# In[6]:


state, info = env.reset()
action = env.action_space.sample()


# In[7]:

path_buffer_inference = 'inference/' if args['inference'] else ''

for path in [path_buffer, path_buffer + path_buffer_inference, path_models]:
    if not os.path.exists(path):
        os.makedirs(path)

buffer = ReplayBuffer(buffer_size=buffer_size, input_shape=(env.observation_space.shape[0],), 
                      n_actions=env.action_space.shape[0], memmap_loc=path_buffer + path_buffer_inference)
mode = 'r+' if os.path.isfile(path_buffer + path_buffer_inference + 'pulse_memory.npy') else 'w+'
pulse_memory = np.memmap(path_buffer + path_buffer_inference + 'pulse_memory.npy', dtype=float, shape=(buffer_size, record_length), mode=mode)

# In[8]:

if os.path.isfile(path_models + 'policy.pt'):
    agent = SoftActorCritic.load(env, path_models)
else:
    agent = SoftActorCritic(env, lr=lr, gamma=gamma, batch_size=batch_size, learning_starts=learning_starts, gradient_steps=gradient_steps, buffer_size=buffer_size, buffer=buffer, 
                            device='cuda' if torch.cuda.is_available() else 'cpu')
print('Agent made.')
    
# In[9]:

if not load or args['inference']:
    buffer.erase()
    pulse_memory[:] = 0.

# In[10]:


userdata = {'agent': agent,
            'env': env,
            'state': state,
            'action': action,
            'buffer': buffer,
            'pulse_memory': pulse_memory,
            'learning_starts': learning_starts, 
            'path_models': path_models,
            'greedy': args['inference'],
            'channel': channel,
            'omega': omega,
            'penalty': penalty,
            'set_pars_msg': set_pars_msg,
            'subscribe_acknowledge_msg': subscribe_acknowledge_msg,
            'trigger_msg': trigger_msg,
            'acknowledge_msg': acknowledge_msg,
            'adc_range': adc_range,
            'dac_range': dac_range,
            'Ib_range': Ib_range,
            'env_steps': env_steps,
            'inference_steps': inference_steps,
            'log_reward': log_reward,
            'inv_reward': False,
            'ph_amp': 0.,
            'steps_per_episode': steps_per_episode,
            'timer': 0,
            'tpa_in_state': tpa_in_state,
            'sweep': generate_sweep(nmbr_dac=12, nmbr_bias=10) if sweep else None,
            'testpulse_interval': testpulse_interval,
            'tau': tau,
            'cph': 1e-5,
           }

client = connect_mqtt(broker, port, client_id, username, password, userdata = userdata)


# In[11]:


subscribe(client, subscribe_acknowledge_msg['topic'])
subscribe(client, trigger_msg['topic'])
subscribe(client, acknowledge_msg['topic'])


# In[12]:


client.on_message = receive_as_control


# In[13]:


channel_info = {"SubscribeToChannel": [channel]}
result = client.publish(subscribe_channel_msg['topic'], json.dumps(channel_info))
check(result)


# In[14]:


# userdata['greedy'] = args['inference']


# In[15]:


client.loop_forever()

print('DONE')

# In[ ]:




