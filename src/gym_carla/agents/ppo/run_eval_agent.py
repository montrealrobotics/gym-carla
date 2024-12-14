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

def rollout_agent(seed, env_id, town, port, max_steps, num_episodes, num_scenarios, num_vehicles, model_path, eval_save_path, scenarios_path=None):
    
    os.makedirs(eval_save_path + "/collision", exist_ok=True)
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
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # policy = PpoPolicy.load(model_path)[0]
    policy = PpoPolicy(env.observation_space, env.action_space, distribution_kwargs={"action_dependent_std": True}).to(device)
    policy.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
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
            ret = info["final_info"]["episode"]["r"]
            episodic_rewards.append(ret)
            episodic_lens.append(ep_len)

            if info["collision"]:
                print("Collision!")
                collision_scenarios.append(info["vehicle_history"])
                save_video(obs, [0], [ep_len], 1, eval_save_path + "/collision", prefix=f"ep_{ep}")\
        
        ep += 1
        
    
    np.save(f"{eval_save_path}/test_episodic_return.npy", np.array(episodic_rewards))
    np.save(f"{eval_save_path}/test_episodic_lens.npy", np.array(episodic_lens))
    if len(collision_scenarios) > 0:
        with open(f"{eval_save_path}/test_crashes.pkl", 'wb') as f:
            pkl.dump(collision_scenarios, f)
    print("Saved all files!")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    save_path = HydraConfig.get().runtime.output_dir
    model_path = Path(save_path).joinpath("policy.ppo_model")
    test_collision_scenarios = Path(save_path).joinpath("test_collision_scenarios.pkl") if cfg.test_on_collisions else None

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

    assert (cfg.num_test_scenarios == 0) != (cfg.num_test_episodes == 0), "Either evaluate on N episodes, or collect N crash scenarios"
    assert ((cfg.num_test_scenarios + cfg.num_test_episodes) == 0) != (test_collision_scenarios is None), "Either evaluate on given collision scenarios or rollout episodes"

    print(OmegaConf.to_yaml(cfg))
    rollout_agent(seed=cfg.seed, env_id=cfg.env_id, town=cfg.town, port=port,
                                max_steps=cfg.num_test_steps, num_episodes=cfg.num_test_episodes, num_scenarios=cfg.num_test_scenarios, num_vehicles=cfg.num_vehicles,
                                model_path=model_path, eval_save_path=save_path, scenarios_path=test_collision_scenarios)

if __name__ == "__main__":
    """
    python eval_agent.py --multirun hydra.sweep.dir=./results/hyperparam_experiment num_steps=1024 agent.learning_rate=0.0003  agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=65536 num_vehicles=25 seed=0 gpu_ids=[0] num_test_episodes=30 num_test_steps=1024
    """
    main()
