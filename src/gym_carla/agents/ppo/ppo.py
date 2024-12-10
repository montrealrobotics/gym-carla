import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from gym_carla.agents.birdview.birdview_wrapper import BirdviewWrapper
from gym_carla.envs.misc import CarlaDummVecEnv
from gym_carla.agents.ppo.ppo_policy import PpoPolicy
from gym_carla.envs.misc import CarlaDummVecEnv, save_video
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
import time
import logging

log = logging.getLogger(__name__)


def make_env(
    env_name="carla-bev-v0",
    number_of_vehicles=50,
    number_of_walkers=0,
    display_size=128,
    max_past_step=2,
    delta_past_step=5,
    dt=0.1,
    discrete=False,
    discrete_acc=[-3.0, 0.0, 3.0],
    discrete_steer=[-0.2, 0.0, 0.2],
    continuous_accel_range=[-1.0, 1.0],
    continuous_steer_range=[-0.3, 0.3],
    ego_vehicle_filter="vehicle.audi.a2",
    port=4000,
    town="Town03",
    task_mode="random",
    max_time_episode=1024,
    max_waypt=12,
    obs_range=32,
    lidar_bin=0.25,
    d_behind=12,
    out_lane_thres=2.0,
    desired_speed=8,
    max_ego_spawn_times=200,
    display_route=True,
    seed=None,
    headless=True,
):
    """Loads train and eval environments."""
    env_params = {
        "number_of_vehicles": number_of_vehicles,
        "number_of_walkers": number_of_walkers,
        "display_size": display_size,  # screen size of bird-eye render
        "max_past_step": max_past_step,  # the number of past steps to draw
        "delta_past_step": delta_past_step,
        "dt": dt,  # time interval between two frames
        "discrete": discrete,  # whether to use discrete control space
        "discrete_acc": discrete_acc,  # discrete value of accelerations
        "discrete_steer": discrete_steer,  # discrete value of steering angles
        "continuous_accel_range": continuous_accel_range,  # continuous acceleration range
        "continuous_steer_range": continuous_steer_range,  # continuous steering angle range
        "ego_vehicle_filter": ego_vehicle_filter,  # filter for defining ego vehicle
        "port": port,  # connection port
        "town": town,  # which town to simulate
        "task_mode": task_mode,  # mode of the task, [random, roundabout (only for Town03)]
        "max_time_episode": max_time_episode,  # maximum timesteps per episode
        "max_waypt": max_waypt,  # maximum number of waypoints
        "obs_range": obs_range,  # observation range (meter)
        "lidar_bin": lidar_bin,  # bin size of lidar sensor (meter)
        "d_behind": d_behind,  # distance behind the ego vehicle (meter)
        "out_lane_thres": out_lane_thres,  # threshold for out of lane
        "desired_speed": desired_speed,  # desired speed (m/s)
        "max_ego_spawn_times": max_ego_spawn_times,  # maximum times to spawn ego vehicle
        "display_route": display_route,  # whether to render the desired route
        "seed": seed,
        "headless": headless,
    }

    env = gym.make(env_name, params=env_params)
    env = BirdviewWrapper(env)
    return env


def run_single_experiment(cfg, seed, save_path, port):
    exp_name = os.path.basename(__file__)[: -len(".py")]
    cfg.agent.batch_size = int(cfg.num_envs * cfg.num_steps)
    cfg.agent.minibatch_size = int(cfg.agent.batch_size // cfg.agent.num_minibatches)
    print("batch size: ", cfg.agent.batch_size)
    print("Minibatch size set as :", cfg.agent.minibatch_size)
    num_iterations = cfg.total_timesteps // cfg.agent.batch_size
    run_name = f"{cfg.env_id}__{exp_name}__{cfg.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])), #TODO: vars cfg prolly needs adjustment
    )

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    # device = "cpu"

    # TODO: eventually we want many envs!!
    # enforcing that max steps are more than num steps here
    env = CarlaDummVecEnv([lambda env_name=env_name: make_env(env_name=env_name, town=env_town, port=port, seed=seed, max_time_episode=max_steps, number_of_vehicles=num_vehicles) for env_name, env_town, port, max_steps, num_vehicles  in [(cfg.env_id, cfg.town, port, cfg.num_steps-1, cfg.num_vehicles)]])
    # env = DummyVecEnv([make_env(env_name=cfg.env_id, town=cfg.town)])
    
    agent = PpoPolicy(env.observation_space, env.action_space, distribution_kwargs=cfg.agent.distribution_kwargs).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=cfg.agent.learning_rate, eps=1e-5)

    obs = {}
    for k, s in env.observation_space.spaces.items():
        obs[k] = torch.zeros((cfg.num_steps, env.num_envs,) + s.shape).to(device)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs) + env.action_space.shape).to(device)
    mus = torch.zeros((cfg.num_steps, cfg.num_envs) + env.action_space.shape).to(device)
    sigmas = torch.zeros((cfg.num_steps, cfg.num_envs) + env.action_space.shape).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # torch.backends.cudnn.deterministic = cfg.torch_deterministic
    env.action_space.seed(cfg.seed)
    env.observation_space.seed(cfg.seed)
    env.seed(cfg.seed)

    global_step = 0
    start_time = time.time()

    kl_early_stop = 0
    t_train_values = 0.0

    collision_scenarios = []
    episodic_rewards_list = []
    episodic_lens_list = []
    next_obs = env.reset()
    for iteration in range(1, num_iterations + 1):
        
        next_done = torch.zeros(env.num_envs).to(device)
        print("Iteration:", iteration)
        # Annealing the rate if instructed to do so.
        if cfg.agent.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * cfg.agent.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        print("Collecting experience...")
        ep_start_idx = [0]
        collision_scenario_idx = []
        collision_scenario_lens = []
        ep_lens = []
        collision_scenario = False
        for step in range(0, cfg.num_steps):
            global_step += cfg.num_envs
            for k, v in next_obs.items():
                obs[k][step] = torch.Tensor(v).to(device)
            dones[step] = next_done
            if next_done:
                # Do this with probability p
                if len(collision_scenarios) > cfg.min_collision_scenes and random.random() < cfg.p:
                    log.info(f"Reset to collision scenario. Step: {global_step}")
                    next_obs = env.reset(vehicle_positions=random.choice(collision_scenarios))
                    collision_scenario = True
                    collision_scenario_idx.append(step)
                else:
                    collision_scenario = False
                    next_obs = env.reset()

            with torch.no_grad():
                action, value, log_prob, mu, sigma, _ = agent.forward(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            mus[step] = mu
            sigmas[step] = sigma
            logprobs[step] = log_prob

            next_obs, reward, next_done, infos = env.step(action.cpu().numpy())
            rewards[step] = torch.Tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(next_done).to(device)

            print("Step:", step, "  Iteration:", iteration)
            
            
            for info in infos:
                if(next_done or step == (cfg.num_steps-1)):
                    if "episode" in info["final_info"]:
                        ret = info["final_info"]["episode"]["r"]
                        ep_len = info["final_info"]["episode"]["l"]
                        ep_lens.append(int(ep_len))
                        if collision_scenario:
                            collision_scenario_lens.append(int(ep_len))
                        if step < cfg.num_steps-1:
                            ep_start_idx.append(step)
                        
                        episodic_rewards_list.append(ret)
                        episodic_lens_list.append(ep_len)
                        if info["collision"] and not collision_scenario:
                            log.info(f"Collision scenarios: {len(collision_scenarios)}. Step: {global_step}")
                            collision_scenarios.append(info["vehicle_history"])
                        print(
                            f"global_step={global_step}, episodic_return={ret}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", ret, global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", ep_len, global_step
                        )
                        writer.add_scalar(
                            "charts/average_reward_in_episode", ret/ep_len, global_step
                        )

        env.clean()

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.forward_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + cfg.agent.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + cfg.agent.gamma * cfg.agent.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values
        
        #TODO : double check 
        # if cfg.save_video:
        #     save_video(obs['birdeye'], ep_start_idx, ep_lens, cfg.save_last_n, save_path)
        #     if len(collision_scenario_idx) > 0:
        #         save_video(obs['birdeye'], collision_scenario_idx, collision_scenario_lens, 5, save_path + "/collision")

        # flatten the batch
        b_obs = {}
        for k, s in obs.items():
            # obs[k] = np.zeros((args.num_steps, env.num_envs,) + s.shape, dtype=s.dtype)
            b_obs[k] = obs[k].reshape((-1,) + obs[k].shape[2:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_mus = mus.reshape((-1,) + env.action_space.shape)
        b_sigmas = sigmas.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.agent.batch_size)
        clipfracs = []
        print("Training...")
        for epoch in range(cfg.agent.update_epochs):
            print("Epoch:", epoch)
            np.random.shuffle(b_inds)
            for start in range(0, cfg.agent.batch_size, cfg.agent.minibatch_size):
                end = start + cfg.agent.minibatch_size
                mb_inds = b_inds[start:end]

                newvalue, newlogprob, entropy_loss, distribution = agent.evaluate_actions(
                    {k: b_obs[k][mb_inds] for k in b_obs.keys()}, b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # with torch.no_grad():
                #     # calculate approx_kl http://joschu.net/blog/kl-approx.html
                #     old_approx_kl = (-logratio).mean()
                #     approx_kl = ((ratio - 1) - logratio).mean()
                #     clipfracs += [((ratio - 1.0).abs() > cfg.agent.clip_coef).float().mean().item()]
                
                with torch.no_grad():
                    old_distribution = agent.action_dist.proba_distribution(
                        b_mus[mb_inds], b_sigmas[mb_inds])
                    old_approx_kl = (-logratio).mean()
                    approx_kl = torch.distributions.kl_divergence(old_distribution.distribution, distribution).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.agent.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                if cfg.agent.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.agent.clip_coef, 1 + cfg.agent.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.agent.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.agent.clip_coef,
                        cfg.agent.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss + cfg.agent.ent_coef * entropy_loss + v_loss * cfg.agent.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.agent.max_grad_norm)
                optimizer.step()

            if cfg.agent.target_kl is not None and approx_kl > cfg.agent.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if cfg.save_model:
        model_path = f"{save_path}/policy.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
        np.save(f"{save_path}/episodic_rewards.npy", np.array(episodic_rewards_list))
        np.save(f"{save_path}/episodic_lens.npy", np.array(episodic_lens_list))
        
        # from cleanrl_utils.evals.ppo_eval import evaluate

        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=Agent,
        #     device=device,
        #     gamma=args.gamma,
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

    env.close()
    writer.close()
    
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    print(">>> Storing outputs in: ", save_path)
    os.makedirs(save_path, exist_ok=True) # TODO: where we'll store results. we need to decide on which stats.
    os.makedirs(save_path + "/collision", exist_ok=True)
    
    result_file = Path(save_path).joinpath("episodic_rewards.npy")
    if not(result_file.is_file()): # if results aren't there already
        
        num_gpus = len(cfg.gpu_ids)
        gpu_id_idx = HydraConfig.get().job.num % num_gpus
        gpu_id = cfg.gpu_ids[gpu_id_idx]
        print("gpu id:", gpu_id)
        print("---------------")
        port = (gpu_id + 4)*1000
        print("port:", port)
        
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        run_single_experiment(cfg, cfg.seed, save_path, port)
        print("Experiment done!")
        
    else:
        print("There are already save results for this hyperparam config. Terminating.")

if __name__ == "__main__":
    run_experiment()