import cryoenv.cryosig as cs
import numpy as np
import time
from scipy.constants import e
from tqdm.auto import trange
import matplotlib.pyplot as plt

np.random.seed(0)
use_sampler = False

if use_sampler:
    pars = cs.sample_parameters()
    print(pars)
    det = cs.DetectorModule(**pars)

else:
    det = cs.DetectorModule()

det.set_control(dac=[-.75], Ib=[-.9], norm=True)
det.wait(5)
det.trigger(er=0., tpa=10, verb=True)
det.plot_event()
det.plot_nps(only_sum=False)


# det.print_noise_parameters()
#
# rew = -det.rms * det.tpa / det.ph
#
# print('Reward: {}'.format(rew))
