from gym.envs.registration import register

name = 'hyperdopamine'

register(
    id='myEnv-v0',
    entry_point='hyperdopamine.interfaces.myEnv:myEnv',
    max_episode_steps=250,
)
