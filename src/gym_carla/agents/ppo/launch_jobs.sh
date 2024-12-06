#!/bin/bash

echo "Launching jobs with hydra ..."

sudo pkill -f "python"
# python ppo.py --multirun hydra.sweep.dir=./results/hyperparam_experiment num_steps=256,512 agent.learning_rate=0.0003,0.00003,0.000003  agent.num_minibatches=8,16,32 agent.update_epochs=10 total_timesteps=131072 num_vehicles=25 seed=0,1,2,3,4
python ppo.py --multirun hydra.sweep.dir=./results/hyperparam_experiment num_steps=1024 agent.learning_rate=0.0003,0.00003,0.000003  agent.num_minibatches=8,16,32 agent.update_epochs=10 total_timesteps=131072 num_vehicles=25 seed=0,1,2,3,4