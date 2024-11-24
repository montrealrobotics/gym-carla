from gym.envs.registration import register

register(
    id='carla-v0',
    entry_point='gym_carla.envs:CarlaEnv',
)

register(
    id='carla-bev-v0',
    entry_point='gym_carla.envs:CarlaBEVEnv',
)