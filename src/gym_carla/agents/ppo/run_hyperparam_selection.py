# go through all result files under an experiment name
import os
from pathlib import Path
import numpy as np
import yaml
from functools import partial
import itertools
import sys

"""
This selects hyperparameters based on mean average episodic reward.
"""

def get_average_episodic_reward(seeds_dir: Path):

    seeds_dir_list = os.listdir(seeds_dir)
    avg_episodic_reward = []

    for s in seeds_dir_list:

        p_reward = Path(seeds_dir).joinpath(s).joinpath(f"train_episodic_rewards.npy")
        p_lens = Path(seeds_dir).joinpath(s).joinpath(f"train_episodic_lens.npy")
        episodic_reward_arr = np.array(np.load(p_reward,  allow_pickle=True))
        episodic_len_arr = np.array(np.load(p_lens,  allow_pickle=True))

        print(type(episodic_reward_arr))
        print("avg: ", episodic_reward_arr / episodic_len_arr)
        avg_episodic_reward.append(episodic_reward_arr / episodic_len_arr)

    return avg_episodic_reward
    
def get_episodic_reward(seeds_dir: Path):

    seeds_dir_list = os.listdir(seeds_dir)
    episodic_reward = []

    for s in seeds_dir_list:

        p_reward = Path(seeds_dir).joinpath(s).joinpath(f"train_episodic_rewards.npy")
        episodic_reward_arr = np.array(np.load(p_reward,  allow_pickle=True))

        episodic_reward.append(episodic_reward_arr)

    return episodic_reward

def hyperparams_list_to_dict(hyperparams_strs: list):

    hyperparams_strs = sorted(hyperparams_strs)
    hyperparam_dict = {}
    for i in range(len(hyperparams_strs)):
        hyperparam_name, vals_str = hyperparams_strs[i].split("=")
        vals = [s for s in vals_str.split(",")]
        hyperparam_dict[hyperparam_name] = vals

    return hyperparam_dict

def generate_hyperparam_paths(hyperparams: dict, base_results_path: str):
    
    product_values = itertools.product(*[v if isinstance(v, (list, tuple)) else [v] for v in hyperparams.values()])
    perms = [dict(zip(hyperparams.keys(), values)) for values in product_values]
    paths = []
    for p in perms:
        curr_path = Path(base_results_path)
        for key, val in p.items():
            if(key == "agent.learning_rate"):
                val = float(val)
            part = key + "-" + str(val)
            curr_path = curr_path.joinpath(part)
        paths.append(curr_path)

    return paths


def mean_episodic_reward_aross_runs(seeds_dir: Path):

    episodic_rewards_runs = get_episodic_reward(seeds_dir=seeds_dir)
    min_len = min(len(i) for i in episodic_rewards_runs)
    episodic_rewards_runs = [ep_r[:min_len] for ep_r in episodic_rewards_runs]
    episodic_rewards_runs_arr = np.array(episodic_rewards_runs)

    return np.mean(episodic_rewards_runs_arr, axis=0)

def select_best_hyperparams(experiment_dir_path: str):

    multi_run_fp = Path(experiment_dir_path).joinpath("multirun.yaml")
    with open(multi_run_fp) as stream:
        try:
            multirun_d = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    hyperparams_strs = multirun_d["hydra"]["overrides"]["task"]
    exclude_keys = multirun_d["hydra"]["job"]["config"]["override_dirname"]["exclude_keys"]

    def exclude_hyperparam(hp):
        for hp_exclude in exclude_keys:
            if(hp_exclude in hp):
                return False
        return True
    
    hyperparams_strs = list(filter(exclude_hyperparam, hyperparams_strs))
    hyperparams = hyperparams_list_to_dict(hyperparams_strs)
    hyperparam_paths = generate_hyperparam_paths(hyperparams, experiment_dir_path)
    
    best_hp = None
    best_m = -np.inf
    for hp in hyperparam_paths:
        
        print("Processing:", hp)
        avg = np.mean(mean_episodic_reward_aross_runs(hp))
        if(avg > best_m):
            best_m = avg
            best_hp = hp
        print("----------")
    return best_hp, best_m


if __name__ in "__main__":
    best_hp, best_m = select_best_hyperparams(sys.argv[1])
    print("Best hyperparameter config:", best_hp)
    print("Average (across episodes) Mean Episodic Reward (across runs):", best_m)


