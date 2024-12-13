import torch
import numpy as np
import pickle as pkl
import os
from gym_carla.agents.ppo.ppo import make_env
from gym_carla.agents.ppo.ppo_policy import PpoPolicy
from gym_carla.envs.misc import CarlaDummVecEnv, save_video
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path

"""
This runs a given policy in "model_path" in carla and gathers crash scenarios along with episodic returns and videos
"""
#TODO: add ability to gather a minimum number of collisions.

def gather_collision_scenarios(seed, env_id, town, port, max_steps, num_episodes, num_vehicles, model_path, eval_save_path):
    
    os.makedirs(eval_save_path + "/collision", exist_ok=True)
    env = CarlaDummVecEnv(
        [
            lambda env_name=env_id: make_env(
                env_name=env_name,
                town=town,
                port=port,
                seed=seed,
                max_time_episode=max_steps-1,
                number_of_vehicles=num_vehicles,
            )
        ]
    )

    policy = PpoPolicy.load(model_path)[0]
    policy = policy.eval()

    obs = torch.zeros((max_steps, 1,) + env.observation_space.spaces['birdeye'].shape)
    collision_scenarios = []

    episodic_rewards = []
    episodic_lens = []
    for ep in range(num_episodes):
        print(f"Episode {ep+1}")
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
            ret = info["final_info"]["episode"]["r"]
            episodic_rewards.append(ret)
            episodic_lens.append(ep_len)

            if info["collision"]:
                print("Collision!")
                collision_scenarios.append(info["vehicle_history"])
                save_video(obs, [0], [ep_len-1], 1, eval_save_path + "/collision", prefix=f"ep_{ep}")
        
    
    np.save(f"{eval_save_path}/test_episodic_return.npy", np.array(episodic_rewards))
    np.save(f"{eval_save_path}/test_episodic_lens.npy", np.array(episodic_lens))
    if len(collision_scenarios) > 0:
        with open(f"{eval_save_path}/test_collision_scenarios.npy", 'wb') as f:
            pkl.dump(collision_scenarios, f)
    print("Saved all files!")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    save_path = HydraConfig.get().runtime.output_dir
    model_path = Path(save_path).joinpath("policy.ppo_model")

    print(">>> Storing outputs in: ", save_path)
    print(">>> Reading in model from: ", model_path)

    num_gpus = len(cfg.gpu_ids)
    gpu_id_idx = HydraConfig.get().job.num % num_gpus
    gpu_id = cfg.gpu_ids[gpu_id_idx]
    print("gpu id:", gpu_id)
    print("---------------")
    port = (gpu_id + 4)*1000
    print("port:", port)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(OmegaConf.to_yaml(cfg))
    gather_collision_scenarios(seed=cfg.seed, env_id=cfg.env_id, town=cfg.town, port=port,
                                max_steps=cfg.num_test_steps, num_episodes=cfg.num_test_episodes, num_vehicles=cfg.num_vehicles,
                                model_path=model_path, eval_save_path=save_path)

if __name__ == "__main__":
    """
    python eval_agent.py --multirun hydra.sweep.dir=./results/hyperparam_experiment num_steps=1024 agent.learning_rate=0.0003  agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=65536 num_vehicles=25 seed=0 gpu_ids=[0] num_test_episodes=30 num_test_steps=1024
    """
    main()
