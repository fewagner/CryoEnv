#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.path.append("/users/felix.wagner/pipfolder/lib/python3.8/site-packages/")
sys.path.append("/users/felix.wagner/cryoenv")

import gymnasium as gym
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from tqdm.auto import tqdm, trange
from cryoenv.mqtt import SoftActorCritic, ReturnTracker, HistoryWriter, CryoWorldModel, generate_sweep, augment_pars, double_tes

import pdb 
import os 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--buffer_save_path', type=str, default='/users/felix.wagner/cryoenv/tests/buffers', help='the path to save the buffer and model')
    parser.add_argument('--pars_saved_path', type=str, default='/users/felix.wagner/cryoenv/tests/saved_pars', help='the path to save the buffer and model')
    parser.add_argument('--detector', type=str, default='li1p', help='which detector, either of li1p li1l li2p')
    parser.add_argument('--version', type=int, default=0, help='the version of the trained model')
    parser.add_argument('--rnd_seed', type=int, default=0, help='the random seed for numpy')
    parser.add_argument('--double_tes', type=bool, default=False, help='are two TES on the absorber')
    args = vars(parser.parse_args())
    
    # In[3]:

    warnings.simplefilter('ignore')
    gym.logger.set_level(40)


    # In[4]:


    name_load = args['detector']
    version = args['version']
    rseed = args['rnd_seed']
    buffer_save_path = args['buffer_save_path'] + '/'
    pars_saved_path = args['pars_saved_path']
    buffer_size = 1400
    
    path_models = '{}{}_v{}/models/'.format(buffer_save_path, name_load, version)
    path_data = '{}{}_v{}/data/'.format(buffer_save_path, name_load, version)
    
    for p in [path_models[:-8], path_models[:-1], path_data[:-1]]:
        try:
            os.mkdir(p)
        except OSError as error:
            print(error)
    
    with open("{}/{}_pars_cryoenv.pkl".format(pars_saved_path, name_load),"rb") as fh:
        pars_load = pickle.load(fh)
        
    tries = 0

    while True:

        if tries > 10:
            raise AssertionError

        add_pars = {
            'store_raw': True,
            'max_buffer_len': buffer_size,
            'tpa_queue': [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'pileup_prob': 0.,
            'tp_interval': 10,
            'dac_range': (0., 5.), 
            'Ib_range': (0.5, 5.), 
            'adc_range': (-10., 10.),
            'rseed': rseed
                    }

        np.random.seed(rseed)

        if args['double_tes']:
            pars_load = double_tes(pars_load)

        aug_pars = augment_pars(pars_load, **add_pars)

        env = gym.make('cryoenv:cryoenv-sig-v0',
                           omega=0.01,
                           sample_pars=False,
                           log_reward=False,
                           rand_start=True,
                           relax_time=60,
                           tpa_in_state=True,
                           div_adc_by_bias=True,
                           pars=aug_pars,
                           render_mode='human',
                           )

        # check if transition is reachable

        env.detector.set_control(dac=np.ones(env.nheater), Ib=np.ones(env.ntes), norm=True)

        for i in range(10):
            env.detector.wait(5)

        try:
            for i in range(env.ntes):  # assumes TES are the first components!
                assert env.detector.Rt[i](env.detector.T[0,i]) > env.detector.Rs[i], 'transition of TES {} not reachable'.format(i)
            print('All transistions reachable, continuing')
            break
        except AssertionError:
            rseed += 1000
            tries += 1
            print('Resampling parameters, new rseed: {}'.format(rseed))
            

    obs, info = env.reset()


    # In[11]:

    model = SoftActorCritic(env, policy = "GaussianPolicy", critic = "QNetwork", lr=3e-4, buffer_size=buffer_size, learning_starts=0,
                            batch_size=16, gamma=.99, gradient_steps=20, grad_clipping=.5, tau=0.005, 
                            device='cuda' if torch.cuda.is_available() else 'cpu')


    # In[12]:


    tracker = ReturnTracker()
    writer = HistoryWriter()


    # In[15]:


    model.learn(episodes = 40, 
                episode_steps = 60, 
                two_pbars=True,
                tracker=tracker,
                writer=writer)


    # In[18]:


    # save buffer

    all_buffers = ['state_memory', 'next_state_memory', 'action_memory', 'reward_memory', 'terminal_memory', 'buffer_counter', 'buffer_total']
    if hasattr(model.buffer, 'trajectory_idx'):
        all_buffers.append('trajectory_idx')

    for name_buffer in all_buffers:
        np.save(path_data + name_buffer + '.npy', eval('model.buffer.{}'.format(name_buffer)))


    # In[20]:


    model.save(path_models)


    # In[23]:


    import json

    with open('{}{}_v{}/info.txt'.format(buffer_save_path, name_load, version), 'w') as fh:
        for key, value in aug_pars.items(): 
            fh.write("'{}': {}\n".format(key, value))

