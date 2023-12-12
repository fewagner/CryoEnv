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
from cryoenv.mqtt import SoftActorCritic, ReturnTracker, HistoryWriter, CryoWorldModel, generate_sweep, augment_pars

from datasets import load_dataset, DatasetDict, Dataset
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments
from dataclasses import dataclass
import os

from cryoenv.envs import CryoEnvSigWrapper

import argparse

from inference_utils import DecisionTransformerGymDataCollator, TrainableDT, get_action


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_sac', type=str, default='/users/felix.wagner/cryoenv/tests/buffers', help='the path to the buffer and model')
    parser.add_argument('--buffer_save_path_inf_sac', type=str, default='/users/felix.wagner/cryoenv/tests/buffers_inf_sac', help='the path to store the sac inference buffer')
    parser.add_argument('--buffer_save_path_inf_dt', type=str, default='/users/felix.wagner/cryoenv/tests/buffers_inf_dt', help='the path to store the dt inference buffer')
    parser.add_argument('--path_checkpoint', type=str, default='/scratch-cbe/users/felix.wagner/rltests/output_40m/checkpoint-90000', help='the path to the dt checkpoint')
    
    parser.add_argument('--detector', type=str, default='li1p', help='which detector, either of li1p li1l li2p')
    parser.add_argument('--version', type=int, default=0, help='the version of the trained model')
    
    parser.add_argument('--do_sac', default=False, action='store_true', help='do inference with the sac')
    parser.add_argument('--do_dt', default=False, action='store_true', help='do inference with the dt')
    
    args = vars(parser.parse_args())

    torch.cuda.is_available()

    warnings.simplefilter('ignore')
    gym.logger.set_level(40)

    
    name_load = args['detector']
    version = args['version']
    rseed = int(version)
    buffer_size = 2800

    path_models = '{}{}_v{}/models/'.format(args['path_sac'], name_load, version)

    buffer_save_path_inf_sac = args['buffer_save_path_inf_sac']
    buffer_save_path_inf_dt = args['buffer_save_path_inf_dt']
    path_inf_sac = '{}/{}_v{}/data/'.format(buffer_save_path_inf_sac, name_load, version)
    path_inf_dt = '{}/{}_v{}/data/'.format(buffer_save_path_inf_dt, name_load, version)


    for p in [path_inf_sac[:-6], path_inf_sac[:-1], path_inf_dt[:-6], path_inf_dt[:-1], ]:
        try:
            os.mkdir(p)
        except OSError as error:
            print(error)

            
    with open("/users/felix.wagner/cryoenv/tests/saved_pars/{}_pars_cryoenv.pkl".format(name_load),"rb") as fh:
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
            'dac_range': (0., 5.) if name_load in ['li1p', 'li2p'] else (0., 10.),  # this was changed
            'Ib_range': (0.5, 5.) if name_load in ['li1p', 'li2p'] else (0.1, 3.),  # this was changed
            'adc_range': (-10., 10.),
            'rseed': rseed,
            'tau_cap': np.array([1. if int(version) < 75 else 20.]),
                    }

        np.random.seed(rseed)

        aug_pars = augment_pars(pars_load, scale=0.1 if version < 10 else 0.2, **add_pars)

#         env = gym.make('cryoenv:cryoenv-sig-v0',
#                            omega=0.01,
#                            log_reward=False,
#                            rand_start=True,
#                            relax_time=60,
#                            tpa_in_state=True,
#                            div_adc_by_bias=True,
#                            pars=aug_pars,
#                            render_mode='human',
#                            rand_tpa=False,
#                            )
        
        env = CryoEnvSigWrapper(omega=.0,  # this has to be zero for inference (unspoiled rewards)
                        weighted_reward=True,
                        log_reward=False,
                        rand_start=True,
                        relax_time=60,
                        tpa_in_state=True,
                        div_adc_by_bias=True,
                        pars=aug_pars,
                        render_mode='human',
                        rand_tpa=False,)
        

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
    
    # # Soft Actor Critic

    if args['do_sac']:

        model = SoftActorCritic.load(env, path='/users/felix.wagner/cryoenv/tests/buffers/{}_v{}/models/'.format(name_load, version),  
                                     device='cuda' if torch.cuda.is_available() else 'cpu')


        obs, _ = env.reset(clear_buffer=True)
        model.policy.eval()
        returns = 0
        for i in trange(60):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action.flatten())
            returns += reward
            if terminated or truncated:
                obs, _ = env.reset()

        env.detector.write_buffer(path_inf_sac + 'buffer')


    # # Decision Transformer

    if args['do_dt']:
    

        dataset_train = load_dataset("pandas", data_files="/users/felix.wagner/cryoenv/tests/saved_pars/transformer_dataset_train.pkl")['train']

        collator = DecisionTransformerGymDataCollator(dataset_train)

        model = TrainableDT.from_pretrained(args['path_checkpoint'])


        # thats crucial!
        model.config.max_length = 60

        
        model = model.to("cpu")
        device = "cpu"

        scale = collator.scale # 1000.0  # normalization for rewards/returns
        TARGET_EXPLORE = -0.024 if name_load == 'li1p' else -0.08 if name_load == 'li1l' else -0.02  # -0.3
        TARGET_EXPLOIT = -0.024 if name_load == 'li1p' else -0.08 if name_load == 'li1l' else -0.02 
        # TARGET_EXPLORE = -0.042 if name_load == 'li1p' else -0.1 if name_load == 'li1l' else -0.041  # -0.3
        # TARGET_EXPLOIT = -0.05 if name_load == 'li1p' else -0.1 if name_load == 'li1l' else -0.05 
        TARGET_RETURN = TARGET_EXPLORE * 60 / scale  # collator.target # / scale  # 12000 / scale  # evaluation is conditioned on a return of 12000, scaled accordingly
        print(scale, TARGET_RETURN)

        state_mean = collator.state_mean.astype(np.float32)
        state_std = collator.state_std.astype(np.float32)
        print(state_mean)

        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)
        
        tries = 1

        for try_ in range(tries):

            print('TRY {}/{}'.format(try_, tries))

            episode_return, episode_length = 0, 0

            state, _ = env.reset(clear_buffer=True)

            target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
            states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)

            switched = False

            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
            for t in trange(60):
                actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])

                action = get_action(
                    model,
                    (states - state_mean) / state_std,
                    actions,
                    rewards,
                    target_return,
                    timesteps,
                )
                actions[-1] = action
                action = action.detach().cpu().numpy()

                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated


                cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
                states = torch.cat([states, cur_state], dim=0)
                rewards[-1] = reward

                pred_return = target_return[0, -1] - (reward / scale)
                target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)


                episode_return += reward
                episode_length += 1

                if reward > -0.15 and not switched:
                    target_return += (TARGET_EXPLOIT * 60 / scale - TARGET_RETURN) * (60-t)/60
                    print('Switching in t={}'.format(t))
                    switched = True

                # if target_return[0,-1] > (60-t)*TARGET_EXPLORE/scale/2 and not switched:
                #     target_return[0, :] += (60-t)*(TARGET_EXPLORE/scale) - target_return[0, -1]
                # if target_return[0,-1] > 0 and switched:
                #     target_return[0, :] += (60-t)*(TARGET_EXPLORE/scale) - target_return[0, -1]
                #     switched = False
                #     print('Switching back in t={}'.format(t))

                timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

                if done:
                    break

            if switched:  # np.mean(-np.array(env.detector.buffer_rms[-30:])/np.array(env.detector.buffer_ph[-30:])) > 2 * TARGET_EXPLOIT:
                print('SUCCESS')
                break

            if try_ == tries - 1:
                print('FAILED')


        env.detector.write_buffer(path_inf_dt + 'buffer')
        print('WROTE BUFFERS')



