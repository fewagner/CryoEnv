from ..base._base_policy import Policy
import numpy as np


class EpsilonGreedy(Policy):

    def __init__(self, epsilon=None):
        super(EpsilonGreedy, self).__init__()
        self.epsilon = epsilon

    def _setup(self):
        """
        Setup the policy function.
        """
        pass

    def update(self, epsilon):
        """
        Update the parameters.
        """
        self.epsilon = epsilon

    def predict(self, observation):
        """
        Get an action state value.
        """
        if np.random.uniform() < self.epsilon:
            return self.action_space.sample()
        else:
            return self.value_function.greedy(observation)
