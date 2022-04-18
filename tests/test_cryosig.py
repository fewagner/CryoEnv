import cryoenv.cryosig as cs
import numpy as np
import time
from scipy.constants import e

np.random.seed(0)
use_sampler = False

if use_sampler:
    pars = cs.sample_parameters()
    print(pars)
    det = cs.DetectorModule(**pars)

else:
    det = cs.DetectorModule()

det.set_control(dac=[-.5], Ib=[-.9], norm=True)
det.wait(5)
det.trigger(er=0., tpa=0.001, verb=True)
det.plot_event()
det.plot_nps()

print('Reward: {}'.format(-det.rms*det.tpa/det.ph))

det.print_noise_parameters()

print(det.pileup_t0, det.pileup_prob, det.pileup_er)