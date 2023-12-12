from gymnasium.envs.registration import register

register(
    id='cryoenv-v0',
    entry_point='cryoenv.envs:CryoEnv_v0',
)

register(
    id='cryoenv-discrete-v0',
    entry_point='cryoenv.envs:CryoEnvDiscrete_v0',
)

register(
    id='cryoenv-continuous-v0',
    entry_point='cryoenv.envs:CryoEnvContinuous_v0',
)

register(
    id='cryoenv-sig-v0',
    entry_point='cryoenv.envs:CryoEnvSigWrapper',
)

__all__ = ['agents',
           'envs',
           'cryosig']
