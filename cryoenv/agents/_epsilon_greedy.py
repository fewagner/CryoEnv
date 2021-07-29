from ..base._base_policy import Policy
import numpy as np


class EpsilonGreedy(Policy):

    def __init__(self, epsilon=None):
        super(EpsilonGreedy, self).__init__()
        self.epsilon = epsilon
        self.was_greedy = False

    # ------------------------------------------------
    # own
    # ------------------------------------------------

    # ------------------------------------------------
    # overwrite parent
    # ------------------------------------------------

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
        self.was_greedy = np.random.uniform() < self.epsilon
        if self.was_greedy:
            return self.value_function.greedy(observation)
        else:
            return self.action_space.sample()
