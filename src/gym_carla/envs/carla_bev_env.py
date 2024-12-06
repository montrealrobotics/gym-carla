#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
import logging

import gym
from gym import spaces
from gym.utils import seeding
import carla

from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *


log = logging.getLogger(__name__)


class CarlaBEVEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
    # parameters
    self.display_size = params['display_size']
    if type(self.display_size) == int:
      self.display_size = (self.display_size, self.display_size)
    self.max_past_step = params['max_past_step']
    self.delta_past_step = params['delta_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.task_mode = params['task_mode']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.d_behind = params['d_behind']
    self._port = params['port']
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self._headless = params['headless'] if 'headless' in params else False
    self._seed = params['seed']

    # Destination
    if params['task_mode'] == 'roundabout':
      self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
    else:
      self.dests = None

    # action and observation spaces
    self.discrete = params['discrete']
    self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
    self.n_acc = len(self.discrete_act[0])
    self.n_steer = len(self.discrete_act[1])
    if self.discrete:
      self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
    else:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], 
      params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
      params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
    observation_space_dict = {
      'birdeye': spaces.Box(low=0, high=255, shape=(self.display_size[0], self.display_size[1], 3), dtype=np.uint8),
      'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
      }
    self.observation_space = spaces.Dict(observation_space_dict)

    # Connect to carla server and get world object
    print('connecting to Carla server...')
    self._client = carla.Client('localhost', self._port)
    self._client.set_timeout(10.0)
    self._tm = self._client.get_trafficmanager(self._port+1500)
    self._tm_port = self._port+1500
    if self._seed:
      self._tm.set_random_device_seed(self._seed)
      random.seed(self._seed)
      np.random.seed(self._seed)
    self._world = self._client.load_world(params['town'])
    print('Carla server connected!')

    # Set weather
    self._world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get spawn points
    self._vehicle_spawn_points = list(self._world.get_map().get_spawn_points())
    self._vehicles = []
    self._vehicles_history = []
    self._walker_spawn_points = []
    for i in range(self.number_of_walkers):
      spawn_point = carla.Transform()
      loc = self._world.get_random_location_from_navigation()
      if (loc != None):
        spawn_point.location = loc
        self._walker_spawn_points.append(spawn_point)

    # Create the ego vehicle blueprint
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length
    self.collision_bp = self._world.get_blueprint_library().find('sensor.other.collision')

    # Set fixed simulation step for synchronous mode
    self.settings = self._world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Disable rendering if not using camera
    self.settings.no_rendering_mode = True

    self._world.apply_settings(self.settings)

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0
    
    # Initialize the renderer
    self._init_renderer()

  def reset(self, vehicle_positions=None):
    # Clear sensor objects  
    self.collision_sensor = None

    # Delete sensors, vehicles and walkers
    self._clear_all_actors(['sensor.other.collision', 'vehicle.*', 'controller.ai.walker', 'walker.*'])
    self._vehicles = []
    self.ego = None
    self._vehicles_history = []

    # Disable sync mode
    self._set_synchronous_mode(False)

    # Spawn surrounding vehicles and ego
    spawned_vehicle_positions = {}
    if vehicle_positions:
      for time_step in reversed(vehicle_positions):
        for key, pos in time_step.items():
          if key != "ego":
            spawn_point = carla.Transform(carla.Location(*pos[:3]), carla.Rotation(*pos[3:]))
            vehicle, loc = self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4])
            if vehicle:
              self._vehicles.append(vehicle)
              spawned_vehicle_positions[vehicle.id] = loc
            else:
              log.error(f"Could not spawn vehicle at {pos}")

        pos = time_step["ego"]
        spawn_point = carla.Transform(carla.Location(*pos[:3]), carla.Rotation(*pos[3:]))
        vehicle, loc = self._try_spawn_ego_vehicle_at(spawn_point)
        if vehicle:
          log.info("Collision scenario!")
          spawned_vehicle_positions["ego"] = loc
          break
        else:
          log.error(f"Could not spawn ego at {pos}")
          self._vehicles = []
          self._clear_all_actors(['vehicle.*'])
      if self.ego is None:
        self.reset()

    if self.ego is None: # Havent spawned ego. Lets sample from initial distribution.
      random.shuffle(self._vehicle_spawn_points)
      count = self.number_of_vehicles
      if count > 0:
        for spawn_point in self._vehicle_spawn_points:
          vehicle, loc = self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4])
          if vehicle:
            self._vehicles.append(vehicle)
            spawned_vehicle_positions[vehicle.id] = loc
            count -= 1
          if count <= 0:
            break
      while count > 0:
        vehicle, loc = self._try_spawn_random_vehicle_at(random.choice(self._vehicle_spawn_points), number_of_wheels=[4])
        if vehicle:
          self._vehicles.append(vehicle)
          spawned_vehicle_positions[vehicle.id] = loc
          count -= 1
      
      ego_spawn_times = 0
      while True:
        if ego_spawn_times > self.max_ego_spawn_times:
          self.reset()

        if self.task_mode == 'random':
          transform = random.choice(self._vehicle_spawn_points)
        if self.task_mode == 'roundabout':
          self.start=[52.1+np.random.uniform(-5,5),-4.2, 178.66] # random
          # self.start=[52.1,-4.2, 178.66] # static
          transform = set_carla_transform(self.start)
        vehicle, loc = self._try_spawn_ego_vehicle_at(transform)
        if vehicle:
          spawned_vehicle_positions["ego"] = loc
          break
        else:
          ego_spawn_times += 1
          time.sleep(0.1)
    
    for veh in self._vehicles:
      veh.set_autopilot(True, tm_port=self._tm_port)

    # Spawn pedestrians
    random.shuffle(self._walker_spawn_points)
    count = self.number_of_walkers
    if count > 0:
      for spawn_point in self._walker_spawn_points:
        if self._try_spawn_random_walker_at(spawn_point):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_walker_at(random.choice(self._walker_spawn_points)):
        count -= 1

    # Get actor locations
    self._world.tick()
    self._vehicles_history.append(spawned_vehicle_positions)

    # Add collision sensor
    self.collision_sensor = self._world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
    self.collision_sensor.listen(lambda event: get_collision_hist(event))
    def get_collision_hist(event):
      impulse = event.normal_impulse
      intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
      self.collision_hist.append(intensity)
      if len(self.collision_hist)>self.collision_hist_l:
        self.collision_hist.pop(0)
    self.collision_hist = []

    # Update timesteps
    self.time_step=0
    self.reset_step+=1

    # Enable sync mode
    self._set_synchronous_mode(True)

    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    return self._get_obs()
  
  def step(self, action):
    if self._terminal():
      log.warning("Stepping terminated env")

    # Record new vehicle locations every delta steps
    if (self.time_step + 1) % self.delta_past_step == 0:
      self._vehicles_history.append(self._get_vehicle_transforms())
    while len(self._vehicles_history) > self.max_past_step:
      self._vehicles_history.pop(0)

    # Calculate acceleration and steering
    if self.discrete:
      acc = self.discrete_act[0][action//self.n_steer]
      steer = self.discrete_act[1][action%self.n_steer]
    else:
      acc = action[0]
      steer = action[1]

    # Convert acceleration to throttle and brake
    if acc > 0:
      throttle = np.clip(acc/3,0,1)
      brake = 0
    else:
      throttle = 0
      brake = np.clip(-acc/8,0,1)

    # Apply control
    act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
    self.ego.apply_control(act)

    self._world.tick()

    # route planner
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # state information
    info = {
      'waypoints': self.waypoints,
      'vehicle_front': self.vehicle_front,
      'vehicle_history': self._vehicles_history,
      'collision': len(self.collision_hist) > 0,
    }
    
    # Update timesteps
    self.time_step += 1
    self.total_step += 1

    return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def render(self, mode):
    pass

  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = self._world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
    return bp

  def _init_renderer(self):
    """Initialize the birdeye view renderer.
    """
    if not self._headless:
      pygame.init()
      self.display = pygame.display.set_mode(
      (self.display_size[0] * 2, self.display_size[1]),
      pygame.HWSURFACE | pygame.DOUBLEBUF)

    pixels_per_meter = self.display_size[0] / self.obs_range
    birdeye_params = {
      'target_size': PixelDimensions(*self.display_size),  
      'pixels_per_meter': pixels_per_meter,
      'crop_type': BirdViewCropType.FRONT_AND_REAR_AREA,
      'render_lanes_on_junctions': True,
    }
    self.birdeye_render = BirdViewProducer(self._client, **birdeye_params)

  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self._world.apply_settings(self.settings)
    self._tm.set_synchronous_mode(synchronous)

  def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
    """Try to spawn a surrounding vehicle at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    blueprint = self._create_vehicle_bluepprint('vehicle.audi.a2', number_of_wheels=number_of_wheels) #vehicle.*
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle, location = self._try_spawn_actor_robust(blueprint, transform)
    if vehicle:
      return vehicle, location
    return False, None
  
  def _try_spawn_actor_robust(self, blueprint, transform):
    vehicle = self._world.try_spawn_actor(blueprint, transform)
    if vehicle:
      return vehicle, [transform.location.x, transform.location.y, transform.location.z, 0.0, transform.rotation.yaw, 0.0]
    spawn_point = carla.Transform(carla.Location(transform.location.x, transform.location.y, 0.2753), transform.rotation)
    vehicle = self._world.try_spawn_actor(blueprint, spawn_point)
    if vehicle:
      return vehicle, [transform.location.x, transform.location.y, 0.2753, 0.0, transform.rotation.yaw, 0.0]
    spawn_point = carla.Transform(carla.Location(transform.location.x, transform.location.y, max(transform.location.z + 0.05, 0.1)), transform.rotation)
    vehicle = self._world.try_spawn_actor(blueprint, spawn_point)
    if vehicle:
      return vehicle, [transform.location.x, transform.location.y, max(transform.location.z + 0.05, 0.1), 0.0, transform.rotation.yaw, 0.0]
    spawn_point = carla.Transform(carla.Location(transform.location.x, transform.location.y, 0.35), transform.rotation)
    vehicle = self._world.try_spawn_actor(blueprint, spawn_point)
    if vehicle:
      return vehicle, [transform.location.x, transform.location.y, 0.35, 0.0, transform.rotation.yaw, 0.0]
    return None, None

  def _try_spawn_random_walker_at(self, transform):
    """Try to spawn a walker at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    walker_bp = random.choice(self._world.get_blueprint_library().filter('walker.*'))
    # set as not invencible
    if walker_bp.has_attribute('is_invincible'):
      walker_bp.set_attribute('is_invincible', 'false')
    walker_actor = self._world.try_spawn_actor(walker_bp, transform)

    if walker_actor is not None:
      walker_controller_bp = self._world.get_blueprint_library().find('controller.ai.walker')
      walker_controller_actor = self._world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
      # start walker
      walker_controller_actor.start()
      # set walk to random point
      walker_controller_actor.go_to_location(self._world.get_random_location_from_navigation())
      # random max speed
      walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
      return True
    return False

  def _try_spawn_ego_vehicle_at(self, transform):
    """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    vehicle = None
    # Check if ego position overlaps with surrounding vehicles
    overlap = False
    # for actor in self._vehicles_history[-1].values():
    #   actor_center = np.array(actor[:2])
    #   ego_center = np.array([transform.location.x, transform.location.y])
    #   dis = np.linalg.norm(actor_center - ego_center)
    #   if dis > 8:
    #     continue
    #   else:
    #     overlap = True
    #     break

    if not overlap:
      vehicle, location = self._try_spawn_actor_robust(self.ego_bp, transform)

    if vehicle is not None:
      self.ego = vehicle
      return True, location
      
    return False, None
  
  def _get_vehicle_transforms(self):
    veh_transforms = {}
    new_vehicles = []
    for veh in self._vehicles:
      if veh.is_alive:
        new_vehicles.append(veh)
        transform = veh.get_transform()
        veh_transforms[veh.id] = [transform.location.x, transform.location.y, transform.location.z, 0.0, transform.rotation.yaw, 0.0]
    
    self._vehicles = new_vehicles

    transform = self.ego.get_transform()
    veh_transforms["ego"] = [transform.location.x, transform.location.y, transform.location.z, 0.0, transform.rotation.yaw, 0.0]
    return veh_transforms
  
  def _get_obs(self):
    """Get the observations."""
    ## Birdeye rendering
    self.birdeye_render.set_waypoints(self.waypoints)
    birdeye = self.birdeye_render.produce(
        agent_vehicle=self.ego  # carla.Actor (spawned vehicle)
    )
    birdeye_rgb = BirdViewProducer.as_rgb(birdeye).astype(np.uint8)

    # Display birdeye image
    if not self._headless:
      birdeye_surface = rgb_to_display_surface(birdeye_rgb, self.display_size)
      self.display.blit(birdeye_surface, (0, 0))

    # Display on pygame
    if not self._headless:
      pygame.display.flip() 

    # State observation
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw/180*np.pi
    lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
    delta_yaw = np.arcsin(np.cross(w, 
      np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

    obs = {
      'birdeye': birdeye_rgb,
      'state': state,
    }

    return obs

  def _get_reward(self):
    """Calculate the step reward."""
    # reward for speed tracking
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    r_speed = -abs(speed - self.desired_speed)
    
    # reward for collision
    r_collision = 0
    if len(self.collision_hist) > 0:
      r_collision = -1

    # reward for steering:
    r_steer = -self.ego.get_control().steer**2

    # reward for out of lane
    ego_x, ego_y = get_pos(self.ego)
    dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
    r_out = 0
    if abs(dis) > self.out_lane_thres:
      r_out = -1

    # longitudinal speed
    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)

    # cost for too fast
    r_fast = 0
    if lspeed_lon > self.desired_speed:
      r_fast = -1

    # cost for lateral acceleration
    r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

    r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1

    return r

  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)

    # If collides
    if len(self.collision_hist)>0: 
      return True

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      return True

    # If at destination
    if self.dests is not None: # If at destination
      for dest in self.dests:
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
          return True

    # If out of lane
    dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
    if abs(dis) > self.out_lane_thres:
      return True

    return False

  def _clear_all_actors(self, actor_filters):
    """Clear specific actors."""
    for actor_filter in actor_filters:
      for actor in self._world.get_actors().filter(actor_filter):
        if actor.is_alive:
          if actor.type_id == 'controller.ai.walker':
            actor.stop()
          actor.destroy()
    
  def __exit__(self, exception_type, exception_value, traceback):
    self.close()
  
  def close(self):
    self._clear_all_actors(['sensor.other.collision', 'vehicle.*', 'controller.ai.walker', 'walker.*']) 
    self._world.tick()
    self._set_synchronous_mode(False)
    self._client = None
    self._world = None
    self._tm = None
  
  def clean(self):
    self._clear_all_actors(['sensor.other.collision', 'vehicle.*', 'controller.ai.walker', 'walker.*']) 
    self._world.tick()
    self._set_synchronous_mode(False)
