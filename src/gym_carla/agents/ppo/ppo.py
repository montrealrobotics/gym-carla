import os
import random
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.vec_env import DummyVecEnv

from gym_carla.agents.birdview.birdview_wrapper import BirdviewWrapper
from gym_carla.agents.ppo.ppo_policy import PpoPolicy


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 512 #1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 256 # 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(
    env_name="carla-v0",
    number_of_vehicles=100,
    number_of_walkers=0,
    display_size=256,
    max_past_step=1,
    dt=0.1,
    discrete=False,
    discrete_acc=[-3.0, 0.0, 3.0],
    discrete_steer=[-0.2, 0.0, 0.2],
    continuous_accel_range=[-3.0, 3.0],
    continuous_steer_range=[-0.3, 0.3],
    ego_vehicle_filter="vehicle.lincoln*",
    port=2000,
    town="Town03",
    task_mode="random",
    max_time_episode=500,
    max_waypt=12,
    obs_range=32,
    lidar_bin=0.25,
    d_behind=12,
    out_lane_thres=2.0,
    desired_speed=8,
    max_ego_spawn_times=200,
    display_route=True,
    pixor_size=64,
    pixor=False,
):
    """Loads train and eval environments."""
    env_params = {
        "number_of_vehicles": number_of_vehicles,
        "number_of_walkers": number_of_walkers,
        "display_size": display_size,  # screen size of bird-eye render
        "max_past_step": max_past_step,  # the number of past steps to draw
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
        "pixor_size": pixor_size,  # size of the pixor labels
        "pixor": pixor,  # whether to output PIXOR observation
    }

    gym_spec = gym.spec(env_name)
    env = gym_spec.make(params=env_params)
    env = BirdviewWrapper(env)
    return env


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = "cpu"

    # TODO: eventually we want many envs!!
    env = DummyVecEnv([lambda env_name=env_name: make_env(env_name=env_name) for env_name in ["carla-v0"]])

    agent = PpoPolicy(env.observation_space, env.action_space).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = {}
    for k, s in env.observation_space.spaces.items():
        obs[k] = torch.zeros((args.num_steps, env.num_envs,) + s.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + env.action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    env.seed(args.seed)

    global_step = 0
    start_time = time.time()

    kl_early_stop = 0
    t_train_values = 0.0

    next_obs = env.reset()
    next_done = torch.zeros(env.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        print("Iteration:", iteration)
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        print("Collecting experience...")
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            for k, v in next_obs.items():
                obs[k][step] = torch.Tensor(v).to(device)
            dones[step] = next_done

            with torch.no_grad():
                action, value, log_prob, mu, sigma, _ = agent.forward(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = log_prob

            next_obs, reward, next_done, infos = env.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(next_done).to(device)

            print("Step:", step)
            # if "final_info" in infos:
                # print(">>> if final_info in infos")
                # for info in infos["final_info"]:
                #     print("for info in infos[\"final_info\"]")
            for info in infos:
                if(next_done or step == (args.num_steps-1)):
                    if "episode" in info["final_info"]:
                        ret = info["final_info"]["episode"]["r"]
                        ep_len = info["final_info"]["episode"]["l"]
                        print(
                            f"global_step={global_step}, episodic_return={ret}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", ret, global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", ep_len, global_step
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.forward_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = {}
        for k, s in obs.items():
            # obs[k] = np.zeros((args.num_steps, env.num_envs,) + s.shape, dtype=s.dtype)
            b_obs[k] = obs[k].reshape((-1,) + obs[k].shape[2:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        print("Training...")
        for epoch in range(args.update_epochs):
            print("Epoch:", epoch)
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # print()
                newvalue, newlogprob, entropy, _ = agent.evaluate_actions(
                    {k: b_obs[k][mb_inds] for k in b_obs.keys()}, b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
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

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
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
