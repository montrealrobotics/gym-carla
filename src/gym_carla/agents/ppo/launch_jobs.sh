#!/bin/bash

echo "Launching jobs with hydra ..."

sudo pkill -f "python"

python ppo.py --multirun hydra.sweep.dir=./results/hyperparam_experiment num_steps=1024 agent.learning_rate=0.0003,0.00003,0.000003  agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=2048 num_vehicles=25 seed=0,1 gpu_ids=[0] > run_gpu0_output.out &
python ppo.py --multirun hydra.sweep.dir=./results/hyperparam_experiment num_steps=1024 agent.learning_rate=0.0003,0.00003,0.000003 agent.num_minibatches=32 agent.update_epochs=10 total_timesteps=2048 num_vehicles=25 seed=2,3 gpu_ids=[1] > run_gpu1_output.out &