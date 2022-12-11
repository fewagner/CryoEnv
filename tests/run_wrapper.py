import gym
import warnings
import numpy as np
from tqdm.auto import trange
import argparse
import matplotlib.pyplot as plt
from cryoenv.agents import SAC

warnings.simplefilter('ignore')
gym.logger.set_level(40)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('load_path', type=str, default='sac_cryosig', help='the path to load the trained model from')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the run data')
    parser.add_argument('--n_steps', type=int, default=10, help='the number of steps for the run, ~ 5 steps/sec')
    parser.add_argument('--omega', type=float, default=1e-2, help='the omega regularization value')
    parser.add_argument('--rnd_seed', type=int, default=0, help='the random seed for numpy')
    parser.add_argument('--tpa_queue', type=float, nargs='+', default=[0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='the test pulse amplitudes which are sent')
    parser.add_argument('--pileup_prob', type=float, default=0.02, help='the probability of pile ups')
    parser.add_argument('--plot', action='store_true', help='a plot of the buffer is shown at the end')
    parser.add_argument('--sample', action='store_true', help='sample the environment parameters')
    parser.add_argument('--render', type=str, default=None,
                        help='if the trajectories should be rendered, either human or mpl')
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

    model = SAC.load(env, args['load_path'])

    obs = env.reset()
    returns = 0
    for i in trange(args['n_steps']):
        action = model.predict(obs).flatten().detach().numpy()
        obs, reward, done, info = env.step(action)
        returns += reward
        if args['render'] is not None:
            env.render(args['render'], save_path='{}_{}.png'.format(args['save_path'], i))
        if done:
            obs = env.reset()

    if args['plot']:
        env.detector.plot_buffer()

    if args['save_path'] is not None:
        env.detector.write_buffer(args['save_path'])

        with open(args['save_path'] + '_info.txt', 'w') as f:
            for k, v in zip(args.keys(), args.values()):
                f.write(f"{k}: {v}\n")

    print('Average reward: {}'.format(returns / args['n_steps']))

    env.close()