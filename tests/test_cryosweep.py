import cryoenv.cryosig as cs
import time
import numpy as np
import argparse

np.random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str, default='sweep_cryosig', help='the path to save the trained model')
    parser.add_argument('--dac', type=float, default=-0.92, help='dac value to sweep')
    parser.add_argument('--Ib', type=float, default=-0.8, help='Ib value to sweep')
    parser.add_argument('--not_normed', action='store_false',
                        help='the dac and Ib values are not normed to the intervals -1 to 1')
    parser.add_argument('--which', type=str, default='dac', help='either dac, bias or stable')
    parser.add_argument('--use_sampler', action='store_true', help='sample parameters for the detector simulation')
    parser.add_argument('--from', type=float, default=1, help='start value of the sweep')
    parser.add_argument('--to', type=float, default=-1, help='stop value of the sweep')
    parser.add_argument('--tpa_queue', type=float, nargs='+', default=[0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='the test pulse amplitudes which are sent')
    parser.add_argument('--pileup_prob', type=float, default=0.02, help='the probability of pile ups')
    parser.add_argument('--pulser_scale', type=float, default=.1, help='factor that is multiplied to tpa')
    parser.add_argument('--dac_ramping_speed', type=float, nargs='+', default=[2e-3, ],
                        help='the dac ramping speed in V/sec')
    parser.add_argument('--Ib_ramping_speed', type=float, nargs='+', default=[5e-3, ],
                        help='the Ib ramping speed in muA/sec')
    parser.add_argument('--nmbr_tes', type=int, default=1, help='how many TES are on the absorber')
    parser.add_argument('--sweep_which', type=int, default=0, help='which TES is sweeped')
    parser.add_argument('--plot', action='store_true', help='a plot of the buffer is shown at the end')
    args = vars(parser.parse_args())

    assert args['nmbr_tes'] > 0, ''
    assert args['sweep_which'] >= 0, ''
    assert args['sweep_which'] < args['nmbr_tes'], ''

    # dac sweep
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
        'Ib': np.zeros(args['nmbr_tes']) - np.sum(args["not_normed"]),
        'dac': np.zeros(args['nmbr_tes']) - np.sum(args["not_normed"]),
        'heater_attenuator': np.ones(args['nmbr_tes']),
        'pulser_scale': args['pulser_scale'] * np.ones(args['nmbr_tes']),
        'tes_flag': np.array([True if i < args['nmbr_tes'] else False for i in range(args['nmbr_tes'] + 1)]),
        'heater_flag': np.array([True if i < args['nmbr_tes'] else False for i in range(args['nmbr_tes'] + 1)]),
        'pileup_comp': args['nmbr_tes'] - 1,
        'xi': np.ones(args['nmbr_tes']),
        'i_sq': 2.e-12 * np.ones(args['nmbr_tes']),
        'tes_fluct': 0.0002 * np.ones(args['nmbr_tes']),
        'emi': 2.e-10 * np.ones(args['nmbr_tes']),
        'tau': 10*np.ones(args['nmbr_tes']),
    }

    if args["use_sampler"]:
        pars = cs.sample_parameters(**kwargs)
        print(pars)
        det = cs.DetectorModel(**pars)
    else:
        det = cs.DetectorModel(**kwargs)

    det.clear_buffer()
    start_time = time.time()

    if args["which"] == 'dac':
        det.set_control([args["from"] if i == args['sweep_which'] else 0. - np.sum(args["not_normed"]) for i in
                         range(args['nmbr_tes'])],
                        [args["Ib"] if i == args['sweep_which'] else 0. - np.sum(args["not_normed"]) for i in
                         range(args['nmbr_tes'])],
                        norm=args["not_normed"])
        det.wait(20)
        det.sweep_dac([args['from']], [args['to']],
                      heater_channel=args['sweep_which'], norm=args["not_normed"])

    elif args["which"] == 'bias':
        det.set_control([args["dac"] if i == args['sweep_which'] else 0. - np.sum(args["not_normed"]) for i in
                         range(args['nmbr_tes'])],
                        [args["from"] if i == args['sweep_which'] else 0. - np.sum(args["not_normed"]) for i in
                         range(args['nmbr_tes'])],
                        norm=args["not_normed"])
        det.wait(20)
        det.sweep_Ib([args['from']], [args['to']],
                     tes_channel=args['sweep_which'], norm=args["not_normed"])

    elif args["which"] == 'stable':
        det.set_control([args["dac"] if i == args['sweep_which'] else 0. - np.sum(args["not_normed"]) for i in
                         range(args['nmbr_tes'])],
                        [args["Ib"] if i == args['sweep_which'] else 0. - np.sum(args["not_normed"]) for i in
                         range(args['nmbr_tes'])],
                        norm=args["not_normed"])
        det.wait(20)
        det.send_testpulses(30)

    if args['plot']:
        for i in range(det.nmbr_tes):
            det.plot_buffer(tes_channel=i)
        det.plot_temperatures()
        det.plot_tes()
        for i in range(det.nmbr_tes):
            det.plot_nps(tes_channel=i, only_sum=False)
    det.write_buffer(args["save_path"])

    with open(args['save_path'] + '_info.txt', 'w') as f:
        for k, v in zip(args.keys(), args.values()):
            f.write(f"{k}: {v}\n")
