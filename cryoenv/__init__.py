from gym.envs.registration import register

register(
    id='cryoenv-v0',
    entry_point='cryoenv.envs:CryoEnv',
)
