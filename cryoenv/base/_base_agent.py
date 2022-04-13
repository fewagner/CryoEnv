import pickle

class Agent:

    def __init__(self, env, policy, value_function):
        self.env = env
        self.policy = policy
        self.value_function = value_function

        # get from environment shape of action and state space
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.nmbr_actions = self.action_space.shape[0]
        self.nmbr_observations = self.observation_space.shape[0]

        # define action and observations for policy and value function
        # TODO: maybe move this to it's own method, so this can be implemented by children?
        #self.value_function.define_spaces(self.action_space, self.observation_space)
        #self.policy.define_spaces(self.action_space, self.observation_space, self.value_function)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, observation):
        """
        We get an action for a given observation from the trained agent.
        """
        return self.policy(observation)

    def save(self, path):
        """
        We save the whole agent with pickle.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    # ---------------------------------------------------------------
    # These below need to be implemented in children!
    # ---------------------------------------------------------------

    def learn(self, nmbr_steps, learning_rate, discounting_factor, **kwargs):
        """
        We perform nmbr_steps with the agent on the environment and update the value function and policy.
        """
        raise NotImplementedError
