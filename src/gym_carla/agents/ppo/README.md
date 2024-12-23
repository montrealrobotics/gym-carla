
# Base-training

Base training is the first phase in our pipeline where we train a policy with PPO, with different hyperparameters.

Example command (one for GPU #1 and another for #2):

python ppo.py --multirun exp_name="example_experiment" agent.learning_rate=0.0003,0.00003,0.000003 agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=524288 num_vehicles=40 seed=0,1 gpu_ids=[0] phase=base_train > run_out_gpu0.txt &

python ppo.py --multirun exp_name="example_experiment" agent.learning_rate=0.0003,0.00003,0.000003 agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=524288 num_vehicles=40 seed=0,1 gpu_ids=[1] phase=base_train > run_out_gpu1.txt &

Note 1: If you're going to split jobs among gpus like the above, just be sure to split at seeds not at hyperparameters. This is for hyperparameter recording purposes when we save results.

Note 2: If you change your mind about hyperparameters after getting results, do not remove old ones. Simply add new ones and the program will skip ones where results are already acquired.

Expected output:
Every seed will have a directory that looks like this:
./results/hyperparam_experiment/phase_base_train/reset_False/hyperparam1/hyperparam2/...../seed_num/

Inside this you should see:
- train_collision_videos <<< for videos of collisions
- train_crash_scenarios.pkl <<< a list of any collision scenarios that we want to use later for resets
- train_episodic_rewards.npy <<< a list where for each episode we recorded the cumulative reward
- trai_episodic_lens.npy <<< a list where for each episode we recorded how many steps were spent in it
- policy.ppo_model <<< the trained model saved at the end

You should also see a multirun.yaml file in ./results/hyperparam_experiment/phase_base_train/reset_False. This is crucial for hyperparameter selection.

# Hyperparameter Selection

After base training, we provide a script to perform hyperparameter selection based on episodic rewards.

Example:

python run_hyperparam_selection.py ./results/hyperparam_experiment/phase_base_train/reset_False

or if you want hyperparameter selection after post-training

python run_hyperparam_selection.py ./results/hyperparam_experiment/phase_post_train/reset_True <<< with resets

python run_hyperparam_selection.py ./results/hyperparam_experiment/phase_post_train/reset_False <<< no resets

Expected output:
you'll see the optimal hyperparameter set printed out


# Post-training

Post training is the second phase in our pipeline where we load a base policy trained with PPO along with crashes gathred as it was running, and then from that starting point we would run PPO training with and without resets.


Example:

(1) Resets activated: python ppo.py --multirun exp_name="example_experiment" agent.learning_rate=0.00003  agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=2048 num_vehicles=40 seed=0 gpu_ids=[0] phase=post_train reset=true p=0.5 model_and_data_dir="./results/example_experiment/phase_base_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_vehicles-40/seed\=1" > run_out_gpu0.txt &

(2) No resets: python ppo.py --multirun exp_name="example_experiment" agent.learning_rate=0.00003  agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=2048 num_vehicles=40 seed=0 gpu_ids=[1] phase=post_train reset=false p=0.0 model_and_data_dir="./results/example_experiment/phase_base_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_vehicles-40/seed\=1" > run_out_gpu1.txt &


Note: Make sure that if there is a break "\" before an equal sign in the path for model_and_data_dir


Expected output:

Every seed will have a directory that looks like this:
./results/hyperparam_experiment/post_train/reset_True/hyperparam1/hyperparam2/...../seed_num/

./results/hyperparam_experiment/post_train/reset_False/hyperparam1/hyperparam2/...../seed_num/

Inside each folder (whether reset or non-reset) this you should see:
- train_collision_videos <<< for videos of collisions
- train_crash_scenarios.pkl <<< a list of any collision scenarios that we want to use later for resets
- train_episodic_rewards.npy <<< a list where for each episode we recorded the cumulative reward
- train_episodic_lens.npy <<< a list where for each episode we recorded how many steps were spent in it
- policy.ppo_model <<< the trained model saved at the end



# Evaluation: Global Robustness

In Global Robustness evaluation, we just run the reset post training policy and the non-reset post-training policy, and we record the episodic reward and episodic length of each for later plotting.

Example command:

python run_global_robustness_eval.py --exp-name=example_experiment --eval-seed=20 --num-eval-episodes=10 --policy-seeds-path="./results/example_experiment/phase_post_train/reset_True/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_vehicles-40/p-0.5/" --env-cfg-yaml="../../conf/config.yaml"

# Evaluation: Crash-focussed

In crash focussed evaluation, we use a given base trained policy to gather crash scenarios. Then we use those crash scenarios as launching points (for spawning our agent vehicle and other vehicles), and record episodic reward and length for given reset and no-reset post train policies. 

Example command:

python run_crash_focussed_eval.py --exp-name=example_experiment --eval-seed=20 --num_gen_scenarios=3 --base_policy_seeds_path="./results/example_experiment/phase_base_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_vehicles-40/" --post_reset_policy_seeds_path="./results/example_experiment/phase_post_train/reset_True/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_vehicles-40/p-0.5/" --post_baseline_policy_seeds_path="./results/example_experiment/phase_post_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_vehicles-40/p-0.0/" --env-cfg-yaml="../../conf/config.yaml"