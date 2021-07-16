
class agent:

    def __init__(self, environment, policy, value_function):
        self.environment = environment
        self.policy = policy
        self.value_function = value_function

        # get from environment shape of action and state space
        self.actions = ...  # these are box or discrete objects from OpenAI
        self.observations = ...

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, observation):
        """
        We get an action for a given observation from the trained agent.
        """
        return self.policy(observation)

    def learn(self, nmbr_steps, learning_rate, discounting_factor):
        """
        We perform nmbr_steps with the agent on the environment and update the value function and policy.
        """
        raise NotImplementedError

    def save(self):
        """
        We save the whole agent with pickle.
        """
        pass  # TODO