import numpy as np


class Detectors:

    def __init__(self, pars: dict = None, ):

        if pars is None:
            pars = {
                'G': np.diag(np.ones(2)),
                'Gb': np.ones(2),
                'Tb': 0,
                'Heaters': [1],
                'Rh': np.ones(1),
                'Tes': [0],
                'Rs': np.ones(1),
                'dac_ion': [(0.5, 1)],  # (k, d)
                'Ib_conversion': [(0.5, 1)],  # (k, d)
                'V_conversion': [(5, 5)],  # (k, d)
                'tpa_conversion': [(5, 5)],  # (k, d)
            }

        self.pars = pars

    def fit(self, X, y):

        # fit an approximation model to the X, y datasets

        pass

    def predict(self, tpas, dac, ):

        pass

        ph, rms = ..., ...

        return ph, rms

    def training_step(self, ):

        pass