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
    det = cs.DetectorModel(**pars)

else:
    det = cs.DetectorModel()

det.set_control(dac=[-.5], Ib=[-.9], norm=True)
for _ in range(2):
    det.wait(10)
    det.trigger(er=[0., 0.], tpa=[10], verb=True)
    det.plot_temperatures()
    det.plot_tes()
    det.plot_nps(only_sum=False)


det.print_noise_parameters()
rew = np.sum(-det.rms * det.tpa / det.ph)
print('Reward: {}'.format(rew))
