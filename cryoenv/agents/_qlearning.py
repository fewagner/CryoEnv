from ..base import Agent
import numpy as np
from tqdm.auto import trange


class QLearning(Agent):

    def __init__(self, env, policy, value_function):
        super(QLearning, self).__init__(env=env, policy=policy, value_function=value_function)

    # ------------------------------------------------
    # own
    # ------------------------------------------------

    # ------------------------------------------------
    # overwrite parent
    # ------------------------------------------------

    def learn(self, nmbr_steps, learning_rate, discount_factor, **kwargs):
        """
        We perform nmbr_steps with the agent on the environment and update the value function and policy.
        """

        assert 'max_epsilon' in kwargs and 'min_epsilon' in kwargs, 'You need to put max_epsilon and min_epslon as arguments!'

        obs = self.env.reset()

        for step in trange(nmbr_steps):

            action = self.policy.predict(obs)
            new_obs, reward, done, info = self.env.step(action)

            if self.policy.was_greedy:

                self.value_function.update(action=action,
                                           observation=obs,
                                           new_value=(1 - learning_rate) * self.value_function.predict(action, obs) +
                                                     learning_rate * (reward +
                                                                      discount_factor * self.value_function.predict(self.value_function.greedy(
                                                       new_obs), new_obs)))

            obs = new_obs

            if done == True:
                break

            # Cutting down on exploration by reducing the epsilon
            self.policy.update(
                epsilon=(1 - step / nmbr_steps) * kwargs['max_epsilon'] + step / nmbr_steps * kwargs['min_epsilon'])
