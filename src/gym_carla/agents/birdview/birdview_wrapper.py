import gym
import numpy as np


class BirdviewWrapper(gym.Wrapper):
    """Environment wrapper to filter observation channels."""

    def __init__(self, env):
        # self._ev_id = list(env._obs_configs.keys())[0]

        env.observation_space = gym.spaces.Dict(
            {'state': env.observation_space['state'],
             'birdeye': env.observation_space['birdeye']})
        
        super(BirdviewWrapper, self).__init__(env)
    
    def reset(self):
        observation = self.env.reset()
        return self.process_obs(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        obs = self.process_obs(observation)
        return obs, reward, done, info

    @staticmethod
    def process_obs(obs, train=True):
        state = obs['state']
        birdview = obs['birdeye']

        if not train:
            birdview = np.expand_dims(birdview, 0)
            state = np.expand_dims(state, 0)

        obs_dict = {
            'state': state,
            'birdeye': birdview
        }
        return obs_dict