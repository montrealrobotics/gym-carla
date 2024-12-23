# Resets for RL Expert Robustness
A fork of gym-carla: an [OpenAI gym third party environment](https://github.com/openai/gym/blob/master/docs/environments.md) for the [CARLA simulator](http://carla.org/).

## Recommended system
- Ubuntu 20.04
- +32 GB RAM memory
- NVIDIA RTX 3070 / NVIDIA RTX 3080 / NVIDIA RTX 4090


## Installation

### Install CARLA
1. Install [CARLA 0.9.15 release](https://github.com/carla-simulator/carla/releases/tag/0.9.15). 
```
mkdir -p /opt/carla-simulator
cd /opt/carla-simulator
wget https://tiny.carla.org/carla-0-9-15-linux
tar -xvzf carla-0-9-15-linux
rm carla-0-9-15-linux
```
2. Install client library
```
export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
```
If you have previously installed the client library with pip, this will take precedence over the .egg file. You will need to uninstall the previous library first.

### Setup our package
1. Setup conda environment
```
conda create -n env_name python=3.7
conda activate env_name
```

2. Clone this git repo in an appropriate folder
```
git clone https://github.com/montrealrobotics/gym-carla.git
```

3. Enter the repo root folder and install the packages:
```
cd gym-carla
pip install -r requirements.txt
pip install -e .
```

## Usage

### Launch CARLA
```
bash /opt/carla-simulator/CarlaUE4.sh -fps=10 -quality-level=Epic -carla-rpc-port=4000 -RenderOffScreen
```

### Training/Evaluation
Follow instructions in the README [here](src/gym_carla/agents/ppo/README.md)

## Description
1.  We provide a dictionary observation including birdeye view semantic representation (obs['birdeye']) using a customized fork of the repository [carla-birdeye-view](https://github.com/akuramshin/carla-birdeye-view):
<div align="center">
  <img src="figures/new_bev.png" width=50%>
</div>
We also provide a state vector observation (obs['state']) which is composed of lateral distance and heading error between the ego vehicle to the target lane center line (in meter and rad), ego vehicle's speed (in meters per second), and and indicator of whether there is a front vehicle within a safety margin.

2. The termination condition is either the ego vehicle collides, runs out of lane, reaches a destination, or reaches the maximum episode timesteps. Users may modify function _terminal in carla_env.py to enable customized termination condition.

3. The reward is a weighted combination of longitudinal speed and penalties for collision, exceeding maximum speed, out of lane, large steering and large lateral accleration. Users may modify function _get_reward in carla_env.py to enable customized reward function.
