
class policy:

    def __init__(self):
        self.actions = None
        self.observations = None
        self.model = None

    def define_actions(self, actions):
        """
        Defines the actions, these are OpenAI objects.
        """
        self.actions = actions

    def define_observations(self, actions):
        """
        Defines the observations, these are OpenAI objects.
        """
        self.actions = actions

    def _setup_model(self):
        """
        Create the model that approximates the values.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, observation):
        """
        Get an action state value.
        """
        return self.model(observation)

    def update(self, **kwargs):
        """
        Update the model with a given value.
        """
        raise NotImplementedError
