from ..base import ValueFunction
import collections
from itertools import product
import numpy as np
from scipy.interpolate import griddata


class Interpolator(ValueFunction):

    def __init__(self, maxlen=100, initval=0, allowed_overdensity=5, method='nearest'):
        super(Interpolator, self).__init__()
        self.maxlen = maxlen
        self.initval = initval
        self.fill_value = initval
        self.allowed_overdensity = allowed_overdensity
        self.method = method
        self.avg_distance = None
        self.x = collections.deque(maxlen=maxlen)
        self.y = collections.deque(maxlen=maxlen)

    # ------------------------------------------------
    # own
    # ------------------------------------------------

    def _get_close(self, sample):
        """
        Get all indices of samples within a box around a given one.
        """
        x_array = np.array(self.x)
        return np.argwhere(np.max(np.abs(x_array - sample), axis=1) < self.avg_distance)

    # ------------------------------------------------
    # overwrite parent
    # ------------------------------------------------

    def _setup(self):
        """
        Create the model that approximates the values.
        """
        for v in product([-1, 1], repeat=self.nmbr_actions + self.nmbr_observations):
            self.x.append(np.array(v))
        for i in range(2 ** (self.nmbr_actions + self.nmbr_observations)):
            self.y.append(self.initval)
        self.avg_distance = (1 / self.maxlen) ** (1 / (self.nmbr_actions + self.nmbr_observations))

    def greedy(self, observation):
        """
        Return the greedy action for a given observation.
        """

        best_action = self.x[0][:self.nmbr_actions]
        print(self.x, self.y, np.concatenate((self.x[0][:self.nmbr_actions], observation)))
        best_value = griddata(self.x,
                              self.y,
                              np.concatenate((self.x[0][:self.nmbr_actions], observation)),
                              method=self.method,
                              fill_value=self.fill_value)
        print(best_action, best_value)
        for ao in self.x:
            v = griddata(self.x,
                         self.y,
                         np.concatenate((ao[:self.nmbr_actions], observation)),
                         method=self.method,
                         fill_value=self.fill_value)
            if v > best_value:
                best_value = v
                best_action = ao[:self.nmbr_actions]

        return best_action

    def update(self, action, observation, new_value):
        """
        Update the model with a given value.
        """
        close_sample_idx = self._get_close(np.concatenate((action, observation)))
        if len(close_sample_idx) > self.allowed_overdensity:
            del self.x[close_sample_idx[0][0]]
            del self.y[close_sample_idx[0][0]]
        self.x.append(np.concatenate((action, observation)))
        self.y.append(new_value)

    def predict(self, action, observation):
        """
        Get an action state value.
        """
        return griddata(self.x,
                        self.y,
                        np.concatenate((action, observation)),
                        method=self.method,
                        fill_value=self.fill_value)
