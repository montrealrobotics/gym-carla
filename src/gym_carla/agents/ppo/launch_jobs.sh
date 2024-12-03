#!/bin/bash

echo "Launching jobs with hydra ..."
python ppo.py --multirun num_steps=2048 agent.learning_rate=0.003,0.0003,0.00003,0.000003  agent.num_minibatches=16,32,64 agent.update_epochs=10 total_timesteps=100000 num_vehicles=25 seed=0,1,2,3,4