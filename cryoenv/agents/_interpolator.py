
from ..base import ValueFunction
import collections
from itertools import product
import numpy as np
from scipy.interpolate import griddata

class Interpolator(ValueFunction):

    def __init__(self, maxlen=100, initval=0, method='nearest'):
        super(Interpolator, self).__init__()
        self.maxlen = maxlen
        self.initval = initval
        self.method = method
        self.x = collections.deque(maxlen=maxlen)
        self.y = collections.deque(maxlen=maxlen)

    def _setup(self):
        """
        Create the model that approximates the values.
        """
        self.x.append([np.array(v) for v in product([-1, 1], repeat=self.nmbr_actions + self.nmbr_observations)])
        for i in range(self.nmbr_actions + self.nmbr_observations):
            self.y.append(self.initval)

    def greedy(self, observation):
        """
        Return the greedy action for a given observation.
        """

        best_action = self.x[0][:self.nmbr_actions]
        best_value = griddata(self.x,
                        self.y,
                        np.concatenate(self.x[0][:self.nmbr_actions], observation),
                        method=self.method)
        for ao in self.x:
            v = griddata(self.x,
                        self.y,
                        np.concatenate(ao[:self.nmbr_actions], observation),
                        method=self.method)
            if v > best_value:
                best_value = v
                best_action = ao[:self.nmbr_actions]

        return best_action

    def update(self, action, observation, new_value):
        """
        Update the model with a given value.
        """
        self.x.append(np.concatenate(action, observation))
        self.y.append(new_value)

    def predict(self, observation, action):
        """
        Get an action state value.
        """
        return griddata(self.x,
                        self.y,
                        np.concatenate(action, observation),
                        method=self.method)