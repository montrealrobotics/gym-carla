import yaml
from pathlib import Path
import numpy as np
import os
import tyro
from gym_carla.agents.ppo.run_eval_agent import evaluate_policy_on_scenarios, generate_crash_data
from dataclasses import dataclass
import pickle as pkl

@dataclass
class Args:
    eval_seed: int
    base_policy_seeds_path: str
    post_reset_policy_seeds_path: str
    post_baseline_policy_seeds_path: str
    exp_name: str
    env_cfg_yaml: str
    num_gen_scenarios: int


'''
python run_crash_focussed_eval.py --exp-name=example_experiment --eval-seed=20 --num_gen_scenarios=3 --base_policy_seeds_path="./results/example_experiment/phase_base_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_steps-1024/num_vehicles-25" --post_reset_policy_seeds_path="./results/example_experiment/phase_post_train/reset_True/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_steps-1024/num_vehicles-25/p-0.5" --post_baseline_policy_seeds_path="./results/example_experiment/phase_post_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_steps-1024/num_vehicles-25/p-0.0" --env-cfg-yaml="../../conf/config.yaml"
'''

def run_crash_focussed_eval(eval_seed, base_policy_seeds_path, post_reset_policy_seeds_path, post_baseline_policy_seeds_path, eval_save_path, env_cfg_yaml, num_gen_scenarios, port):
    with open(env_cfg_yaml) as stream:
        try:
            env_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    
    base_seed_paths = sorted(os.listdir(base_policy_seeds_path))
    reset_post_seed_paths = sorted(os.listdir(post_reset_policy_seeds_path))
    baseline_post_seed_paths = sorted(os.listdir(post_baseline_policy_seeds_path))
    base_seed_paths.remove("multirun.yaml")

    min_len = min(len(base_seed_paths), len(reset_post_seed_paths), len(baseline_post_seed_paths))
    base_seed_paths = base_seed_paths[:min_len]
    reset_post_seed_paths = reset_post_seed_paths[:min_len]
    baseline_post_seed_paths = baseline_post_seed_paths[:min_len]

    print(">>>>>>>", base_seed_paths, reset_post_seed_paths, baseline_post_seed_paths)
    for base_seed_dir, reset_post_seed_dir, baseline_post_seed_dir in zip(base_seed_paths, reset_post_seed_paths, baseline_post_seed_paths):

        seed_dir = f"base-{base_seed_dir}_reset-{reset_post_seed_dir}_baseline-{baseline_post_seed_dir}"

        base_train_policy_path = Path(base_policy_seeds_path).joinpath(base_seed_dir).joinpath("policy.ppo_model")
        reset_post_policy_path = Path(post_reset_policy_seeds_path).joinpath(reset_post_seed_dir).joinpath("policy.ppo_model")
        baseline_post_policy_path = Path(post_baseline_policy_seeds_path).joinpath(baseline_post_seed_dir).joinpath("policy.ppo_model")
        
        collision_scenarios = generate_crash_data(eval_seed=eval_seed, base_train_policy_path=base_train_policy_path, env_cfg=env_cfg, num_scenarios=num_gen_scenarios, port=port)
        collision_scenarios_dir_path = Path(f"{eval_save_path}/crash_scenarios/{seed_dir}/")
        collision_scenarios_dir_path.mkdir(parents=True, exist_ok=True)
        collision_scenarios_path = collision_scenarios_dir_path.joinpath("test_crash_scenarios.pkl")

        if len(collision_scenarios) > 0:
            with open(collision_scenarios_path, 'wb') as f:
                pkl.dump(collision_scenarios, f)
        
        print("Done with generating crash scenario data. Now evaluating on that crash scenario data.")
        
        reset_episodic_rewards, reset_episodic_lens = evaluate_policy_on_scenarios(eval_seed=eval_seed, post_train_policy_path=reset_post_policy_path, env_cfg=env_cfg, port=port, scenarios_path=collision_scenarios_path)
        baseline_episodic_rewards, baseline_episodic_lens = evaluate_policy_on_scenarios(eval_seed=eval_seed, post_train_policy_path=baseline_post_policy_path, env_cfg=env_cfg, port=port, scenarios_path=collision_scenarios_path)

        Path(f"{eval_save_path}/resets_agent/{seed_dir}").mkdir(parents=True, exist_ok=True)
        Path(f"{eval_save_path}/baseline_agent/{seed_dir}").mkdir(parents=True, exist_ok=True)

        np.save(f"{eval_save_path}/resets_agent/{seed_dir}/test_episodic_return.npy", np.array(reset_episodic_rewards))
        np.save(f"{eval_save_path}/resets_agent/{seed_dir}/test_episodic_lens.npy", np.array(reset_episodic_lens))

        np.save(f"{eval_save_path}/baseline_agent/{seed_dir}/test_episodic_return.npy", np.array(baseline_episodic_rewards))
        np.save(f"{eval_save_path}/baseline_agent/{seed_dir}/test_episodic_lens.npy", np.array(baseline_episodic_lens))

    

def main():
    args = tyro.cli(Args)
    port=4000
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    experiment_config_dict = {  
                            "eval_seed": args.eval_seed,
                            "base_policy_seeds_path": args.base_policy_seeds_path,
                            "post_reset_policy_seeds_path": args.post_reset_policy_seeds_path,
                            "post_baseline_policy_seeds_path": args.post_baseline_policy_seeds_path,
                            "exp_name": args.exp_name,
                            "env_cfg_yaml": args.env_cfg_yaml,
                            "num_gen_scenarios": args.num_gen_scenarios
                            }

    eval_save_path = Path("./results").joinpath(args.exp_name).joinpath("evals").joinpath("crash_focussed_eval")
    eval_save_path.mkdir(parents=True, exist_ok=True)
    run_crash_focussed_eval(
                            eval_seed=args.eval_seed, base_policy_seeds_path=args.base_policy_seeds_path,
                            post_reset_policy_seeds_path=args.post_reset_policy_seeds_path,
                            post_baseline_policy_seeds_path=args.post_baseline_policy_seeds_path, eval_save_path=eval_save_path,
                            env_cfg_yaml=args.env_cfg_yaml, num_gen_scenarios=args.num_gen_scenarios, port=port
                            )
    # with open(eval_save_path.joinpath('eval_config.yaml'), 'w') as file:
    #     yaml.dump(experiment_config_dict, file)
    print("Done crash focussed eval!")

if __name__ == "__main__":
    main()