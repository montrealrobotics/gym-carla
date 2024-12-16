
# Base-training


Example command:

python ppo.py --multirun exp_name="example_experiment" num_steps=1024 agent.learning_rate=0.0003,0.00003 agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=2048 num_vehicles=25 seed=0,1 gpu_ids=[0] phase=base_train > run_out_gpu0.txt &

python ppo.py --multirun exp_name="example_experiment" num_steps=1024 agent.learning_rate=0.0003,0.00003 agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=2048 num_vehicles=25 seed=2,3 gpu_ids=[1] phase=base_train > run_out_gpu1.txt &

Note 1: If you're going to split jobs among gpus like the above, just be sure to split at seeds not at hyperparameters. This is for hyperparameter recording purposes when we save results.

Note 2: If you change your mind about hyperparameters after getting results, do not remove old ones. Simply add new ones and the program will skip ones where results are already acquired.

Expected output:
Every seed will have a directory that looks like this:
./results/hyperparam_experiment/phase_base_train/reset_False/hyperparam1/hyperparam2/...../seed_num/

Inside this you should see:
- collision <<< for videos of collisions
- crash_scenarios.pkl <<< a list of any collision scenarios that we want to use later for resets
- episodic_rewards.npy <<< a list where for each episode we recorded the cumulative reward
- episodic_lens.npy <<< a list where for each episode we recorded how many steps were spent in it
- policy.ppo_model <<< the trained model saved at the end

You should also see a multirun.yaml file in ./results/hyperparam_experiment/phase_base_train/reset_False. This is crucial for hyperparameter selection.

# Hyperparameter Selection

Example:

python run_hyperparam_selection.py ./results/hyperparam_experiment/phase_base_train/reset_False

or if you want hyperparameter selection after post-training

python run_hyperparam_selection.py ./results/hyperparam_experiment/phase_post_train/reset_True <<< with resets

python run_hyperparam_selection.py ./results/hyperparam_experiment/phase_post_train/reset_False <<< no resets

Expected output:
you'll see the optimal hyperparameter set printed out


# Post-training

Example:

(1) Resets activated: python ppo.py --multirun exp_name="example_experiment" num_steps=1024 agent.learning_rate=0.00003  agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=2048 num_vehicles=25 seed=0 gpu_ids=[0] phase=post_train reset=true p=0.5 model_and_data_dir="./results/example_experiment/phase_base_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_steps-1024/num_vehicles-25/seed\=0" > run_out_gpu0.txt &

(2) No resets: python ppo.py --multirun exp_name="example_experiment" num_steps=1024 agent.learning_rate=0.00003  agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=2048 num_vehicles=25 seed=0 gpu_ids=[0] phase=post_train reset=false p=0.0 model_and_data_dir="./results/example_experiment/phase_base_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_steps-1024/num_vehicles-25/seed\=0" > run_out_gpu0.txt &


Note: Make sure that if there is a break "\" before an equal sign in the path for model_and_data_dir


Expected output:

Every seed will have a directory that looks like this:
./results/hyperparam_experiment/post_train/reset_True/hyperparam1/hyperparam2/...../seed_num/

./results/hyperparam_experiment/post_train/reset_False/hyperparam1/hyperparam2/...../seed_num/

Inside each folder (whether reset or non-reset) this you should see:
- collision <<< for videos of collisions
- crash_scenarios.pkl <<< a list of any collision scenarios that we want to use later for resets
- episodic_rewards.npy <<< a list where for each episode we recorded the cumulative reward
- episodic_lens.npy <<< a list where for each episode we recorded how many steps were spent in it
- policy.ppo_model <<< the trained model saved at the end



# Evaluation: Global Robustness

Example command:

python run_global_robustness_eval.py --exp-name=example_experiment --eval-seed=20 --num-eval-episodes=10 --policy-seeds-path="./results/example_experiment/phase_post_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_steps-1024/num_vehicles-25/p-0.0/" --env-cfg-yaml="../../conf/config.yaml"

# Evaluation: Crash-focussed

[WARNING: THIS IS UNDER TESTING]

Example command:

python run_crash_focussed_eval.py --exp-name=example_experiment --eval-seed=20 --num_gen_scenarios=3 --base_policy_seeds_path="./results/example_experiment/phase_base_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_steps-1024/num_vehicles-25" --post_reset_policy_seeds_path="./results/example_experiment/phase_post_train/reset_True/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_steps-1024/num_vehicles-25/p-0.5" --post_baseline_policy_seeds_path="./results/example_experiment/phase_post_train/reset_False/agent.learning_rate-3e-05/agent.num_minibatches-32/agent.update_epochs-10/num_steps-1024/num_vehicles-25/p-0.0" --env-cfg-yaml="../../conf/config.yaml"

