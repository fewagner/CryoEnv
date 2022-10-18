import cryoenv.cryosig as cs
import time
import numpy as np
from scipy.constants import e
import argparse

np.random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str, default='sweep_cryosig', help='the path to save the trained model')
    parser.add_argument('--dac', type=float, default=-0.98, help='dac value to sweep')
    parser.add_argument('--Ib', type=float, default=-0.9, help='Ib value to sweep')
    parser.add_argument('--not_normed', action='store_false', help='the dac and Ib values are not normed to the intervals -1 to 1')
    parser.add_argument('--which', type=str, default='dac', help='either dac, bias or stable')
    parser.add_argument('--use_sampler', action='store_true', help='sample parameters for the detector simulation')
    parser.add_argument('--from', type=float, default=1, help='start value of the sweep')
    parser.add_argument('--to', type=float, default=-1, help='stop value of the sweep')
    parser.add_argument('--tpa_queue', type=float, nargs='+', default=[0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='the test pulse amplitudes which are sent')
    parser.add_argument('--pileup_prob', type=float, default=0.02, help='the probability of pile ups')
    parser.add_argument('--dac_ramping_speed', type=float, nargs='+', default=[2e-3,], help='the dac ramping speed in V/sec')
    parser.add_argument('--Ib_ramping_speed', type=float, nargs='+', default=[5e-3,], help='the Ib ramping speed in muA/sec')
    parser.add_argument('--plot', action='store_true', help='a plot of the buffer is shown at the end')
    args = vars(parser.parse_args())

    # dac sweep
    if args["use_sampler"]:
        pars = cs.sample_parameters(tpa_queue=args["tpa_queue"],
                                    pileup_prob=args["pileup_prob"],
                                    dac_ramping_speed=np.array(args["dac_ramping_speed"]),
                                    Ib_ramping_speed=np.array(args["Ib_ramping_speed"]),
                                    )
        print(pars)
        det = cs.DetectorModule(**pars)

    else:
        det = cs.DetectorModule(tpa_queue=args["tpa_queue"],
                                pileup_prob=args["pileup_prob"],
                                dac_ramping_speed=np.array(args["dac_ramping_speed"]),
                                Ib_ramping_speed=np.array(args["Ib_ramping_speed"]),
                                )

    det.clear_buffer()
    start_time = time.time()

    if args["which"] == 'dac':
        det.set_control([args['from']], [args["Ib"]], norm=args["not_normed"])
        det.wait(5)
        det.sweep_dac([args['from']], [args['to']], norm=args["not_normed"])

    elif args["which"] == 'bias':
        det.set_control([args["dac"]], [args['from']], norm=args["not_normed"])
        det.wait(5)
        det.sweep_Ib([args['from']], [args['to']], norm=args["not_normed"])

    elif args["which"] == 'stable':
        det.set_control([args["dac"]], [args["Ib"]], norm=args["not_normed"])
        det.wait(5)
        det.send_testpulses(500)

    if args['plot']:
        det.plot_buffer()
    det.write_buffer(args["save_path"])

    with open(args['save_path'] + '_info.txt', 'w') as f:
        for k, v in zip(args.keys(), args.values()):
            f.write(f"{k}: {v}\n")
