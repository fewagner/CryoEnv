import gymnasium as gym
import warnings
import numpy as np
from tqdm.auto import trange
import argparse
import matplotlib.pyplot as plt
from cryoenv.agents import SAC

warnings.simplefilter('ignore')
# gym.logger.set_level(40)

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
    parser.add_argument('--pulser_scale', type=float, default=.1, help='factor that is multiplied to tpa')
    parser.add_argument('--dac_ramping_speed', type=float, nargs='+', default=[2e-3, ],
                        help='the dac ramping speed in V/sec')
    parser.add_argument('--Ib_ramping_speed', type=float, nargs='+', default=[5e-3, ],
                        help='the Ib ramping speed in muA/sec')
    parser.add_argument('--nmbr_tes', type=int, default=1, help='how many TES are on the absorber')
    parser.add_argument('--plot', action='store_true', help='a plot of the buffer is shown at the end')
    parser.add_argument('--sample', action='store_true', help='sample the environment parameters')
    parser.add_argument('--render', type=str, default=None,
                        help='if the trajectories should be rendered, either human or mpl')
    args = vars(parser.parse_args())

    assert args['nmbr_tes'] > 0, ''

    np.random.seed(args['rnd_seed'])

    kwargs = {
        'tpa_queue': args["tpa_queue"],
        'pileup_prob': args["pileup_prob"],
        'dac_ramping_speed': args["dac_ramping_speed"] * np.ones(args['nmbr_tes']),
        'Ib_ramping_speed': args["Ib_ramping_speed"] * np.ones(args['nmbr_tes']),
        'C': np.array([5e-5 if i < args['nmbr_tes'] else 5e-4 for i in range(args['nmbr_tes'] + 1)]),
        # first components are TES
        'Gb': 0.005 * np.ones(args['nmbr_tes'] + 1),
        'G': np.array([[(0 if j < args['nmbr_tes'] else 0.001) if i < args['nmbr_tes'] else (
            0.001 if j < args['nmbr_tes'] else 0) for j in range(args['nmbr_tes'] + 1)] for i in
                       range(args['nmbr_tes'] + 1)]),
        'eps': np.array([[((0.99 if i == j else 0) if j < args['nmbr_tes'] else 0.01) if i < args['nmbr_tes'] else (
            0.05 if j < args['nmbr_tes'] else 0.9) for j in range(args['nmbr_tes'] + 1)] for i in
                         range(args['nmbr_tes'] + 1)]),
        'delta': np.array(
            [[(0.98 if i == j else 0) if j < args['nmbr_tes'] else 0.02 for j in range(args['nmbr_tes'] + 1)] for i in
             range(args['nmbr_tes'])]),
        'Rs': 0.035 * np.ones(args['nmbr_tes']),
        'Rh': 10 * np.ones(args['nmbr_tes']),
        'L': 3.5e-07 * np.ones(args['nmbr_tes']),
        'Rt0': 0.2 * np.ones(args['nmbr_tes']),
        'k': 2. * np.ones(args['nmbr_tes']),
        'Tc': 15. * np.ones(args['nmbr_tes']),
        'Ib': np.zeros(args['nmbr_tes']) - 1,
        'dac': np.zeros(args['nmbr_tes']) - 1,
        'heater_attenuator': .1*np.ones(args['nmbr_tes']),
        'pulser_scale': args['pulser_scale'] * np.ones(args['nmbr_tes']),
        'tes_flag': np.array([True if i < args['nmbr_tes'] else False for i in range(args['nmbr_tes'] + 1)]),
        'heater_flag': np.array([True if i < args['nmbr_tes'] else False for i in range(args['nmbr_tes'] + 1)]),
        'pileup_comp': args['nmbr_tes'] - 1,
        'xi': np.ones(args['nmbr_tes']),
        'i_sq': 2.e-12 * np.ones(args['nmbr_tes']),
        'tes_fluct': 0.0002 * np.ones(args['nmbr_tes']),
        'emi': 2.e-10 * np.ones(args['nmbr_tes']),
        'tau': 10*np.ones(args['nmbr_tes']),
        'store_raw': False,
        'max_buffer_len': 1e6,
    }

    env = gym.make('cryoenv:cryoenv-sig-v0',
                   omega=args['omega'],
                   sample_pars=args['sample'],
                   pars=kwargs,
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
        for i in range(env.detector.nmbr_tes):
            env.detector.plot_buffer(tes_channel=i)

    if args['save_path'] is not None:
        env.detector.write_buffer(args['save_path'])

        with open(args['save_path'] + '_info.txt', 'w') as f:
            for k, v in zip(args.keys(), args.values()):
                f.write(f"{k}: {v}\n")

    print('Average reward: {}'.format(returns / args['n_steps']))

    env.close()