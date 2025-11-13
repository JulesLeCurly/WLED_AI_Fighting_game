import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple
import random

from Function.torch.dqn_agent import DQNAgent

from Function.other.All_basic_function import read_yml

import Function.core.Game_function as Game_function

import Function.Vizualizer.Vizualizer as Viz

class Combat1v1Env():
    def __init__(self, strip_length=300, num_players=2, train=False):
        if num_players > 8:
            raise ValueError("Maximum number of players is 8.")
        
        self.is_train = train

        Config = read_yml("Config/Game_env.yml")
        # World parameters
        self.strip_length = strip_length
        self.num_players = num_players
        self.max_steps = Config["max_steps"]




        
        # Generate player IDs
        self.player_ids = [f"p{i+1}" for i in range(num_players)]
        
        # Player colors for visualization (RGB)
        self.player_colors = {
            "p1": [255, 0, 0],      # Red
            "p2": [0, 0, 255],      # Blue
            "p3": [0, 255, 0],      # Green
            "p4": [255, 255, 0],    # Yellow
            "p5": [255, 0, 255],    # Magenta
            "p6": [0, 255, 255],    # Cyan
            "p7": [255, 128, 0],    # Orange
            "p8": [128, 0, 255],    # Purple
        }
        
        # Game parameters
        self.max_hp = Config["max_hp"]
        self.max_mana = Config["max_mana"]
        self.mana_regen = Config["mana_regen"]


        # Action configuration
        self.actions_config = Config["actions_config"]


        self.actions_config_with_name = {}
        for k, v in self.actions_config.items():
            action_params = v.copy()
            action_params.pop('name', None)
            self.actions_config_with_name[v["name"]] = action_params
        
        # Spaces
        self.action_space = spaces.Discrete(len(self.actions_config))


        self.Observation_dict = {
            "my_hp": {"low": 0, "high": self.max_hp},
            "my_mana": {"low": 0, "high": self.max_mana},
            "avg_opp_hp": {"low": 0, "high": self.max_hp},
            "avg_opp_mana": {"low": 0, "high": self.max_mana},
            "num_alive": {"low": 0, "high": num_players},

            "closest_distance_to_opp_1": {"low": 0, "high": strip_length},
            "closest_direction_to_opp_1": {"low": -1, "high": 1},
            "opp_1_hp": {"low": 0, "high": self.max_hp},
            "opp_1_mana": {"low": 0, "high": self.max_mana},

            "closest_distance_to_opp_2": {"low": 0, "high": strip_length},
            "closest_direction_to_opp_2": {"low": -1, "high": 1},
            "opp_2_hp": {"low": 0, "high": self.max_hp},
            "opp_2_mana": {"low": 0, "high": self.max_mana},

            "Randomness_factor": {"low": 0, "high": 1}
        }
        
        self.observation_space = spaces.Box(
            low=np.array([dict_observation_one_ellement["low"] for dict_observation_one_ellement in self.Observation_dict.values()]),
            high=np.array([dict_observation_one_ellement["high"] for dict_observation_one_ellement in self.Observation_dict.values()]),
            dtype=np.float32
        )

        self.agent = DQNAgent(state_size=self.observation_space.shape[0], action_size=self.action_space.n)

        self.visualizer = Viz.Vizualizer("wled")
        self.reset()

    def reset(self):
        self.positions = {player_id: random.randint(0, self.strip_length) for player_id in self.player_ids}

        # Initialize stats
        self.hp = {pid: self.max_hp for pid in self.player_ids}
        self.mana = {pid: self.max_mana for pid in self.player_ids}
        self.alive = {pid: True for pid in self.player_ids}
        self.shield_on_map = [False] * self.strip_length


        self.history = []

        # Step counter
        self.current_step = 0

        # Get observations for all players
        observations = {pid: self._get_observation(pid) for pid in self.player_ids}
        info = self._get_info()
        
        return observations, info
    
    def Episode(self, visualization=False):
        observations, info = self.reset()

        log_player_rewards = {pid: 0 for pid in self.player_ids}
        log_survival_times = {pid: 0 for pid in self.player_ids}

        while self.current_step < self.max_steps:
            observations, rewards, terminated, truncated, info, All_game_actions = self.all_step(observations)
            self.current_step += 1
            

            # LOG for each player and step
            for player_id in self.player_ids:
                log_player_rewards[player_id] += rewards[player_id]
            
                if self.alive[player_id]:
                    log_survival_times[player_id] += 1
        
        if self.is_train:
                
            # 5️⃣ Mise à jour epsilon
            self.agent.decay_epsilon()

            if self.agent.training_step % 100 == 0:
                self.agent.update_target_network()
        if visualization != False:
            if visualization == "Wled":
                self.visualizer.show(All_game_actions, self.positions, self.shield_on_map)
                
            
        return log_player_rewards, log_survival_times, self.current_step, sum(self.alive.values())

    def all_step(self, observations):

        Dict_of_actions = {}
        Dict_of_REAL_actions = {}
        # Get actions
        for player_id in self.player_ids:
            Dict_of_REAL_actions[player_id] = self.agent.select_action(observations[player_id], training=self.is_train)
            Dict_of_actions[player_id] = [self.actions_config[i]["name"] for i, v in enumerate(Dict_of_REAL_actions[player_id]) if v == 1 and i in self.actions_config]

        rewards, terminated, All_game_actions = self.step_game(Dict_of_actions)

        # Get observations
        next_observations = {pid: self._get_observation(pid) for pid in self.player_ids}
        truncated = self.current_step >= self.max_steps
        info = self._get_info()

        # Train if needed
        if self.is_train:
            done = terminated or truncated
            for player_id in self.player_ids:
                
                self.agent.store_transition(
                    state = observations[player_id],
                    action = Dict_of_REAL_actions[player_id],
                    reward = rewards[player_id],
                    next_state = next_observations[player_id],
                    done = not self.alive[player_id] or done
                )

                self.agent.train()

        return observations, rewards, terminated, truncated, info, All_game_actions
    
    def step_game(self, Dict_of_actions):
        prev_hp = {pid: self.hp[pid] for pid in self.player_ids}
        # Execute actions
        rewards = {
            player_id: 0
            for player_id in self.player_ids
        }

        Action_order = [
            "go_left",
            "go_right",
            "random_tp",
            "left_shield",
            "right_shield",
            "punsh",
            "lazer",
            "heal",
        ]

        All_game_actions = {}

        for action in Action_order:
            for player_id in self.player_ids:
                if action in Dict_of_actions[player_id] and self.alive[player_id]:
                    reward, accepted_action = self._execute_action(player_id, action)
                    rewards[player_id] += reward
                    if accepted_action != []:
                        All_game_actions[player_id] = accepted_action[0]
        
        # Regenerate mana for alive players
        for player_id in self.player_ids:
            if self.alive[player_id]:
                self.mana[player_id] = min(self.max_mana, self.mana[player_id] + self.mana_regen)
        
        # Check game end
        alive_count = sum(self.alive.values())
        terminated = alive_count <= 1 or self.current_step >= self.max_steps
        
        # Add damage-based rewards
        for player_id in self.player_ids:
            if self.alive[player_id]:
                damage_dealt = prev_hp[player_id] - self.hp[player_id]
                rewards[player_id] += damage_dealt * 1.1
        
        # Add penalty for taking damage
        for player_id in self.player_ids:
            if self.alive[player_id]:
                damage_taken = prev_hp[player_id] - self.hp[player_id]
                if damage_taken > 0:
                    rewards[player_id] -= damage_taken * 0.5  # Penalty for taking damage

        if terminated:
            # Victory bonus for survivors
            for player_id in self.player_ids:
                if self.alive[player_id]:
                    rewards[player_id] += 100
        
        return rewards, terminated, All_game_actions

    def _get_observation(self, player_id):
        """Get observation for a specific player"""
        if not self.alive[player_id]:
            return np.zeros(len(self.Observation_dict), dtype=np.float32)
        
        # Calculate average opponent HP
        alive_opponents = [pid for pid in self.player_ids if pid != player_id and self.alive[pid]]
        
        avg_opp_hp = np.mean([self.hp[pid] for pid in alive_opponents]) if alive_opponents else 0
        avg_opp_mana = np.mean([self.mana[pid] for pid in alive_opponents]) if alive_opponents else 0

        closest_players = Game_function._get_n_closest_opponents(
            player_id=player_id,
            positions=self.positions,
            alive=self.alive,
            player_ids=self.player_ids,
            strip_length=self.strip_length,
            n=2)

        # Opponent 1
        if len(closest_players) >= 1:
            closest_dist_opp_1, closest_dir_opp_1, closest_pid_opp_1 = closest_players[0]
            opp_1_hp = self.hp[closest_pid_opp_1] if closest_pid_opp_1 is not None else 0
            opp_1_mana = self.mana[closest_pid_opp_1] if closest_pid_opp_1 is not None else 0
        else:
            closest_dist_opp_1, closest_dir_opp_1, closest_pid_opp_1 = self.strip_length / 2, 0, None
            opp_1_hp, opp_1_mana = 0, 0

        if len(closest_players) == 2:
            # FIXED: Opponent 2 (was using [0] twice!)
            closest_dist_opp_2, closest_dir_opp_2, closest_pid_opp_2 = closest_players[1]
            opp_2_hp = self.hp[closest_pid_opp_2] if closest_pid_opp_2 is not None else 0
            opp_2_mana = self.mana[closest_pid_opp_2] if closest_pid_opp_2 is not None else 0
        else:
            closest_dist_opp_2, closest_dir_opp_2, closest_pid_opp_2 = self.strip_length / 2, 0, None
            opp_2_hp, opp_2_mana = 0, 0

        Randomness_factor = random.uniform(0, 1)

        return np.array([
            self.hp[player_id] / self.max_hp, # my_hp
            self.mana[player_id] / self.max_mana, # my_mana
            avg_opp_hp / self.max_hp, # avg_opp_hp
            avg_opp_mana / self.max_mana, # avg_opp_mana
            len(alive_opponents) / (self.num_players - 1), # num_alive

            closest_dist_opp_1 / self.strip_length, # closest_dist_opp_1
            closest_dir_opp_1, # closest_dir_opp_1
            opp_1_hp / self.max_hp, # opp_1_hp
            opp_1_mana / self.max_mana, # opp_1_mana

            closest_dist_opp_2 / self.strip_length, # closest_dist_opp_2
            closest_dir_opp_2, # closest_dir_opp_2
            opp_2_hp / self.max_hp, # opp_2_hp
            opp_2_mana / self.max_mana, # opp_2_mana

            Randomness_factor
        ])

    def _execute_action(self, player_id, action):
        accepted_action = []

        if not self.alive[player_id]:
            return 0
        
        reward = 0

        

        if self.mana[player_id] < self.actions_config_with_name[action]["mana"]:
            return reward - 2, accepted_action  # Invalid action penalty
        else:
            self.mana[player_id] -= self.actions_config_with_name[action]["mana"]

            if action == "left":
                self.positions[player_id] -= 1
                if self.positions[player_id] < 0:
                    self.positions[player_id] += self.strip_length - 1
                accepted_action.append({"name": action})
            elif action == "right":
                self.positions[player_id] += 1
                if self.positions[player_id] >= self.strip_length:
                    self.positions[player_id] = 0
                accepted_action.append({"name": action})
            # Execute attacks
            elif action in ["punsh", "lazer"]:
                damage = self.actions_config_with_name[action]["damage"]
                attack_range = self.actions_config_with_name[action]["range"]
                
                # Find all targets in range
                targets_hit = []
                for target_id in self.player_ids:
                    if target_id != player_id and self.alive[target_id]:
                        dist = Game_function._get_cyclic_distance(
                            self.positions[player_id],
                            self.positions[target_id],
                            self.strip_length
                        )
                        dir = Game_function._get_direction(
                            self.positions[player_id],
                            self.positions[target_id],
                            self.strip_length
                        )
                        shield_blocked = False
                        pos_before_shield = 0
                        for i in range(self.positions[player_id] + dir, self.positions[target_id] + dir, dir):
                            if i < 0:
                                i += self.strip_length
                            if i >= self.strip_length:
                                i -= self.strip_length
                            if self.shield_on_map[i]:
                                shield_blocked = True
                                pos_before_shield = i - dir
                                break
                        if shield_blocked:
                            accepted_action.append({"name": action, "shield_blocked": True, "pos_before_shield": pos_before_shield})
                        elif dist <= attack_range:
                            targets_hit.append(target_id)
                            accepted_action.append({"name": action, "shield_blocked": False})
        
                if targets_hit:
                    # Hit someone!
                    for target_id in targets_hit:
                        self.hp[target_id] -= damage
                        if self.hp[target_id] <= 0:
                            self.alive[target_id] = False
                            accepted_action.append({"name": "death", "Target": target_id})
                            reward += 75  # Kill bonus
                    reward += 10 * len(targets_hit)  # Hit reward
                else:
                    reward -= 20  # Miss penalty
                    
            if action == "left_shield":
                if self.positions[player_id] - 1 < 0:
                    self.shield_on_map[self.strip_length - 1] = True
                else:
                    self.shield_on_map[self.positions[player_id] - 1] = True
                accepted_action.append({"name": action})
            elif action == "right_shield":
                if self.positions[player_id] + 1 >= self.strip_length:
                    self.shield_on_map[0] = True
                else:
                    self.shield_on_map[self.positions[player_id] + 1] = True
                accepted_action.append({"name": action})
            elif action == "heal":
                self.hp[player_id] += self.actions_config_with_name[action]["regen"]
                if self.hp[player_id] > self.max_hp:
                    self.hp[player_id] = self.max_hp
                accepted_action.append({"name": action})
            elif action == "random_tp":
                tp_accepted = False
                while not tp_accepted:
                    new_pos = random.randint(0, self.strip_length - 1)
                    if not self.shield_on_map[new_pos]:
                        tp_accepted = True
                self.positions[player_id] = new_pos
                accepted_action.append({"name": action, "new_pos": new_pos})
            else:
                # Rest
                pass
        

        return reward, accepted_action

    def _get_info(self):
        """Return game state info"""
        return {
            'positions': self.positions.copy(),
            'hp': self.hp.copy(),
            'mana': self.mana.copy(),
            'alive': self.alive.copy(),
            'step': self.current_step,
            'num_alive': sum(self.alive.values())
        }