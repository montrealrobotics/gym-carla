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
        
        self.episodic_return = 0.0
        self.episode_steps = 0.0
    
    def reset(self):
        observation = self.env.reset()
        self.episodic_return = 0.0
        self.episode_steps = 0.0
        return self.process_obs(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        obs = self.process_obs(observation)
        self.episodic_return += reward
        self.episode_steps += 1.0
        
        info["final_info"] = {"episode":{"r": self.episodic_return, "l": self.episode_steps}}
        
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