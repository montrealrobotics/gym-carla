import yaml
from pathlib import Path
import numpy as np
import os
import tyro
from gym_carla.agents.ppo.run_eval_agent import evaluate_policy
from dataclasses import dataclass

@dataclass
class Args:
    eval_seed: int
    policy_seeds_path: str
    exp_name: str
    env_cfg_yaml: str
    num_eval_episodes: int

def run_global_robustness_eval(eval_seed, post_train_policy_seeds_path, eval_save_path, env_cfg_yaml, num_eval_episodes, port):
    with open(env_cfg_yaml) as stream:
        try:
            env_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    seed_paths = os.listdir(post_train_policy_seeds_path)
    for seed_dir in seed_paths:
        post_train_policy_path = Path(post_train_policy_seeds_path).joinpath(seed_dir).joinpath("policy.ppo_model")
        episodic_rewards, episodic_lens = evaluate_policy(eval_seed, post_train_policy_path, env_cfg, num_eval_episodes, port)
        Path(f"{eval_save_path}/{seed_dir}/eval_seed={eval_seed}").mkdir(parents=True, exist_ok=True)
        np.save(f"{eval_save_path}/{seed_dir}/eval_seed={eval_seed}/test_episodic_return.npy", np.array(episodic_rewards))
        np.save(f"{eval_save_path}/{seed_dir}/eval_seed={eval_seed}/test_episodic_lens.npy", np.array(episodic_lens))


def main():
    # python run_eval_agent.py --exp-name=example_experiment --eval-seed=20 --num-eval-episodes=10 --policy-seeds-path="./results/example_experiment/phase_post_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_steps-1024/num_vehicles-25/p-0.0/seed=0/policy.ppo_model" --env-cfg-yaml="../../conf/config.yaml"
    args = tyro.cli(Args)
    port=4000
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    experiment_config_dict = {  
                            "eval_seed": args.eval_seed,
                            "policy_seeds_path": args.policy_seeds_path,
                            "exp_name": args.exp_name,
                            "env_cfg_yaml": args.env_cfg_yaml,
                            "num_eval_episodes": args.num_eval_episodes
                            }

    policy_path_no_exp_folder = "/".join(args.policy_seeds_path.split("results")[1].split("/")[2:])
    eval_save_path = Path("./results").joinpath(args.exp_name).joinpath("evals").joinpath("global_robustness_eval").joinpath(policy_path_no_exp_folder)
    eval_save_path.mkdir(parents=True, exist_ok=True)
    print("Eval safe path: ", eval_save_path)
    
    run_global_robustness_eval(args.eval_seed, args.policy_seeds_path, eval_save_path, args.env_cfg_yaml, args.num_eval_episodes, port)
    with open(eval_save_path.joinpath('eval_config.yaml'), 'w') as file:
        yaml.dump(experiment_config_dict, file)
    print("Done global robustness eval!")

if __name__ == "__main__":
    main()