import torch
import numpy as np
import pickle as pkl
import tyro
import os
from dataclasses import dataclass
from gym_carla.agents.ppo.ppo import make_env
from gym_carla.agents.ppo.ppo_policy import PpoPolicy
from gym_carla.envs.misc import CarlaDummVecEnv, save_video


@dataclass
class Args:
    model_path: str
    """path to model checkpoint file"""
    save_path: str
    """path where to save eval results"""
    seed: int = 1
    """seed of the experiment"""
    env_id: str = "carla-bev-v0"
    """the id of the environment"""
    town: str = "Town03"
    """the id of the Carla town"""
    port: int = 4000
    """the port of the Carla server"""
    max_steps: int = 1024
    """max env steps per episode"""
    num_episodes: int = 10
    """number of episodes to use for rollouts"""
    num_vehicles: int = 25
    """number of vehicles to spawn per episode"""


def main(seed, env_id, town, port, max_steps, num_episodes, num_vehicles, model_path, save_path):
    env = CarlaDummVecEnv(
        [
            lambda env_name=env_id: make_env(
                env_name=env_name,
                town=town,
                port=port,
                seed=seed,
                max_time_episode=max_steps,
                number_of_vehicles=num_vehicles,
            )
        ]
    )

    policy = PpoPolicy.load(model_path)[0]
    policy = policy.eval()

    obs = torch.zeros((max_steps, 1,) + env.observation_space.spaces['birdeye'].shape)
    ep_reward = np.zeros(max_steps)
    sum_rewards = np.zeros(num_episodes)
    collision_scenarios = []

    for ep in range(num_episodes):
        print(f"Episode {ep+1}")
        next_obs = env.reset()

        step = 0
        next_done = False
        while not next_done:
            action, value, log_prob, mu, sigma, _ = policy.forward(next_obs)

            next_obs, reward, next_done, infos = env.step(action.cpu().numpy())
            ep_reward[step] = reward.flatten()
            obs[step] = torch.Tensor(next_obs['birdeye'])

        for info in infos:
            if info["collision"] or True:
                print("Collision!")
                collision_scenarios.append(info["vehicle_history"])
                ep_len = int(info["final_info"]["episode"]["l"])
                print(obs.shape)
                save_video(obs, [0], [ep_len-1], 1, save_path + "/collision", prefix=f"ep_{ep}")
        
        sum_rewards[ep] = np.sum(ep_reward)
    
    
    np.save(f"{save_path}/episodic_return.npy", np.array(sum_rewards))
    if len(collision_scenarios) > 0:
        with open(f"{save_path}/collision_scenarios.npy", 'wb') as f:
            pkl.dump(collision_scenarios, f)
    print("Saved all files!")


if __name__ == "__main__":
    args = tyro.cli(Args)
    os.makedirs(args.save_path + "/collision", exist_ok=True)
    main(args.seed, args.env_id, args.town, args.port, args.max_steps, args.num_episodes, args.num_vehicles, args.model_path, args.save_path)
