import gym
import warnings
import numpy as np
from stable_baselines3 import SAC
from tqdm.auto import tqdm
from cryoenv.envs._utils_stablebaselines import ProgressBarManager
import argparse

warnings.simplefilter('ignore')
gym.logger.set_level(40)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str, default='sac_cryosig', help='the path to save the trained model')
    parser.add_argument('--n_steps', type=int, default=1000, help='the number of steps for the training, ~ 5 steps/sec')
    parser.add_argument('--omega', type=float, default=1e-2, help='the omega regularization value')
    parser.add_argument('--gamma', type=float, default=0.9, help='the discount factor')
    parser.add_argument('--rnd_seed', type=int, default=0, help='the random seed for numpy')
    parser.add_argument('--tpa_queue', type=float, nargs='+', default=[0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='the test pulse amplitudes which are sent')
    parser.add_argument('--pileup_prob', type=float, default=0.02, help='the probability of pile ups')
    parser.add_argument('--gradient_steps', type=int, default=10,
                        help='the number of gradient steps after every evaluation')
    parser.add_argument('--plot', action='store_true', help='a plot of the buffer is shown at the end')
    parser.add_argument('--sample', action='store_true', help='sample the environment parameters')
    args = vars(parser.parse_args())

    np.random.seed(args['rnd_seed'])

    env = gym.make('cryoenv:cryoenv-sig-v0',
                   omega=args['omega'],
                   sample_pars=args['sample'],
                   pars={'store_raw': False,
                         'max_buffer_len': 1e6,
                         'tpa_queue': args["tpa_queue"],
                         'pileup_prob': args["pileup_prob"]},
                   )

    obs = env.reset()

    model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=int(1e6), learning_starts=16,
                batch_size=16, gamma=args['gamma'], train_freq=1, gradient_steps=args['gradient_steps'], )
    with ProgressBarManager(args['n_steps']) as callback:
        model.learn(total_timesteps=args['n_steps'], log_interval=4, callback=callback)
    model.save(args['save_path'])
    if args['plot']:
        env.detector.plot_buffer()

    env.detector.write_buffer(args['save_path'])

    with open(args['save_path'] + '_info.txt', 'w') as f:
        for k, v in zip(args.keys(), args.values()):
            f.write(f"{k}: {v}\n")

    env.close()
