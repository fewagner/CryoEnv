
import gymnasium as gym

class Policy:

    def __init__(self):
        self.action_space = None
        self.observation_space = None

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def define_spaces(self, action_space, observation_space, value_function):
        """
        Defines the action and obsersavtion spaces, these are OpenAI objects.
        """

        assert type(action_space) == gym.spaces.box.Box, 'action_space must be  of type gym.spaces.box.Box!'
        assert type(observation_space) == gym.spaces.box.Box, 'observation_space must be  of type gym.spaces.box.Box!'
        assert all(action_space.low == -1), 'action_space.low must all be -1!'
        assert all(action_space.high == 1), 'action_space.high must all be 1!'
        assert all(observation_space.low == -1), 'observation_space.low must all be -1!'
        assert all(observation_space.high == 1), 'observation_space.high must all be 1!'

        self.action_space = action_space
        self.observation_space = observation_space
        self.nmbr_actions = action_space.shape[0]
        self.nmbr_observations = observation_space.shape[0]
        self.value_function = value_function
        self._setup()

    # ---------------------------------------------------------------
    # These below need to be implemented in children!
    # ---------------------------------------------------------------

    def _setup(self):
        """
        Setup the policy function.
        """
        raise NotImplementedError

    def update(self, **kwargs):
        """
        Update the parameters.
        """
        raise NotImplementedError

    def predict(self, observation):
        """
        Get an action state value.
        """
        raise NotImplementedError