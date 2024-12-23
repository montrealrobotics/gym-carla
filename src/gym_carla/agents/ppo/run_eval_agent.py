import torch
import numpy as np
import pickle as pkl
import os
from gym_carla.agents.ppo.ppo import make_env
from gym_carla.agents.ppo.ppo_policy import PpoPolicy
from gym_carla.envs.misc import CarlaDummVecEnv, save_video


def rollout_agent(seed, env_id, town, port, max_steps, num_episodes, num_scenarios, num_vehicles, model_path, scenarios_path=None):
    
    # eval_save_path="./myvideos"
    # os.makedirs(eval_save_path + "/collision", exist_ok=True)
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

    if scenarios_path:
        with open(scenarios_path, 'rb') as f:
            test_collision_scenarios = pkl.load(f)
    else:
        test_collision_scenarios = []
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PpoPolicy.load(model_path)[0]
    policy.eval()

    obs = torch.zeros((max_steps+1, 1,) + env.observation_space.spaces['birdeye'].shape)
    collision_scenarios = []

    episodic_rewards = []
    episodic_lens = []
    ep = 0
    while True:
        if num_episodes and ep >= num_episodes:
            break
        if num_scenarios and len(collision_scenarios) >= num_scenarios:
            break
        if scenarios_path and ep >= len(test_collision_scenarios):
            break
        print(f"Episode {ep+1}")
        if len(test_collision_scenarios) > 0:
            print(f"Reset to collision scenario")
            next_obs = env.reset(vehicle_positions=test_collision_scenarios[ep])
        else:
            next_obs = env.reset()

        step = 0
        next_done = False
        while not next_done:
            action, value, log_prob, mu, sigma, _ = policy.forward(next_obs)
            next_obs, reward, next_done, infos = env.step(action.cpu().numpy())
            obs[step] = torch.Tensor(next_obs['birdeye'])
            step += 1


        for info in infos:

            ep_len = int(info["final_info"]["episode"]["l"]) #TODO: Need to save episodic length
            # save_video(obs, [0], [ep_len], 1, eval_save_path + "/collision", prefix=f"ep_{ep}")

            ret = info["final_info"]["episode"]["r"]
            episodic_rewards.append(ret)
            episodic_lens.append(ep_len)

            if info["collision"]:
                print("Collision!")
                collision_scenarios.append(info["vehicle_history"])
                # save_video(obs, [0], [ep_len], 1, eval_save_path + "/collision", prefix=f"ep_{ep}")
        print("Gathered ", len(collision_scenarios), "collision scenarios")
        ep += 1
        
    return episodic_rewards, episodic_lens, collision_scenarios
    

#TODO: make num_steps set by command line args
def generate_crash_data(eval_seed, base_train_policy_path, env_cfg, num_scenarios, port):
    _, _, collision_scenarios = rollout_agent(seed=eval_seed, env_id=env_cfg["env_id"], town=env_cfg["town"],
                   port=port, max_steps=env_cfg["num_steps"]-1,
                    num_episodes=0, num_scenarios=num_scenarios, num_vehicles=env_cfg["num_vehicles"],
                    model_path=base_train_policy_path, scenarios_path=None)
    return collision_scenarios

def evaluate_policy(eval_seed, post_train_policy_path, env_cfg, num_eval_episodes, port):
    episodic_rewards, episodic_lens, _ = rollout_agent(seed=eval_seed, env_id=env_cfg["env_id"], town=env_cfg["town"],
                   port=port, max_steps=env_cfg["num_steps"]-1,
                    num_episodes=num_eval_episodes, num_scenarios=0, num_vehicles=env_cfg["num_vehicles"],
                    model_path=post_train_policy_path, scenarios_path=None)
    return episodic_rewards, episodic_lens

def evaluate_policy_on_scenarios(eval_seed, post_train_policy_path, env_cfg, port, scenarios_path):
    episodic_rewards, episodic_lens, _ = rollout_agent(seed=eval_seed, env_id=env_cfg["env_id"], town=env_cfg["town"],
                   port=port, max_steps=env_cfg["num_steps"]-1,
                    num_episodes=0, num_scenarios=0, num_vehicles=env_cfg["num_vehicles"],
                    model_path=post_train_policy_path, scenarios_path=scenarios_path)
    return episodic_rewards, episodic_lens



