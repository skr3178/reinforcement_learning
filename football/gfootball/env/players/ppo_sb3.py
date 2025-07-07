# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Player from Stable-Baselines3 PPO checkpoint.

Example usage with play_game script:
python3 -m gfootball.play_game \
    --players "ppo_sb3:left_players=1,checkpoint=$YOUR_PATH"

This player loads models trained with stable-baselines3 PPO.
"""

from gfootball.env import football_action_set
from gfootball.env import observation_preprocessing
from gfootball.env import player_base
import numpy as np
import torch


class Player(player_base.PlayerBase):
  """An agent loaded from Stable-Baselines3 PPO model checkpoint."""

  def __init__(self, player_config, env_config):
    player_base.PlayerBase.__init__(self, player_config)

    self._action_set = (env_config['action_set']
                        if 'action_set' in env_config else 'default')
    
    # Load the stable-baselines3 model
    checkpoint_path = player_config['checkpoint']
    try:
      from stable_baselines3 import PPO
      self._model = PPO.load(checkpoint_path)
      print(f"Successfully loaded SB3 model from {checkpoint_path}")
    except Exception as e:
      print(f"Error loading SB3 model: {e}")
      raise
    
    # Set model to evaluation mode
    self._model.policy.eval()
    
    # Get stacking configuration
    stacking = 4 if player_config.get('stacked', True) else 1
    self._stacker = ObservationStacker(stacking)
    
    # Store action mapping
    self._action_mapping = football_action_set.action_set_dict[self._action_set]

  def take_action(self, observation):
    assert len(observation) == 1, 'Multiple players control is not supported'

    # Preprocess observation
    observation = observation_preprocessing.generate_smm(observation)
    observation = self._stacker.get(observation)
    
    # Convert to tensor and add batch dimension
    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
    
    # Get action from model
    with torch.no_grad():
      action, _ = self._model.predict(obs_tensor.numpy(), deterministic=True)
    
    # Convert action index to football action
    football_action = self._action_mapping[action]
    return [football_action]

  def reset(self):
    self._stacker.reset()


class ObservationStacker(object):
  """Utility class that produces stacked observations."""

  def __init__(self, stacking):
    self._stacking = stacking
    self._data = []

  def get(self, observation):
    if self._data:
      self._data.append(observation)
      self._data = self._data[-self._stacking:]
    else:
      self._data = [observation] * self._stacking
    return np.concatenate(self._data, axis=-1)

  def reset(self):
    self._data = [] 