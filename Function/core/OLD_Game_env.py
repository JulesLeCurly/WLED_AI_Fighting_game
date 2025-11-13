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
    """Fixed combat environment with proper reward attribution"""
    
    def __init__(self, strip_length=294, num_players=2, train=False):
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
        
        # Player colors
        self.player_colors = {
            "p1": [255, 0, 0],      # Red
            "p2": [0, 0, 255],      # Blue
            "p3": [0, 255, 0],      # Green
            "p4": [255, 255, 0],    # Yellow
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
        
        # Action space: ONE action per step
        self.action_space = spaces.Discrete(len(self.actions_config))
        
        # Observation space (13 features, all normalized)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(13,),  # FIXED: Was 11, but we return 13 values
            dtype=np.float32
        )
        
        # Create ONE shared agent for self-play
        self.agent = DQNAgent(
            state_size=self.observation_space.shape[0], 
            action_size=self.action_space.n,
            learning_rate=0.0003,
            gamma=0.99
        )
        
        self.visualizer = Viz.Vizualizer("wled")
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Random positions, ensuring no overlap
        positions = random.sample(range(self.strip_length), self.num_players)
        self.positions = {self.player_ids[i]: positions[i] for i in range(self.num_players)}
        
        # Initialize stats
        self.hp = {pid: self.max_hp for pid in self.player_ids}
        self.mana = {pid: self.max_mana for pid in self.player_ids}
        self.alive = {pid: True for pid in self.player_ids}
        self.shield_on_map = [False] * self.strip_length
        
        # History for animations
        self.history = []
        self.current_step = 0
        
        # Get observations
        observations = {pid: self._get_observation(pid) for pid in self.player_ids}
        info = self._get_info()
        
        return observations, info
    
    def Episode(self, visualization=False):
        """Run one complete episode"""
        observations, info = self.reset()
        
        log_player_rewards = {pid: 0 for pid in self.player_ids}
        log_survival_times = {pid: 0 for pid in self.player_ids}
        
        while self.current_step < self.max_steps:
            # Get actions (ONE per player)
            actions = {}
            for player_id in self.player_ids:
                if self.alive[player_id]:
                    action_idx = self.agent.select_action(observations[player_id], training=self.is_train)
                    actions[player_id] = action_idx
                else:
                    actions[player_id] = 0  # Default action for dead players
            
            # Execute step
            next_observations, rewards, terminated, truncated, info, All_game_actions = self.step(actions)
            
            # Store transitions for training
            if self.is_train:
                for player_id in self.player_ids:
                    if player_id in observations:  # Only if player was alive
                        self.agent.store_transition(
                            state=observations[player_id],
                            action=actions[player_id],
                            reward=rewards[player_id],
                            next_state=next_observations[player_id],
                            done=(not self.alive[player_id]) or terminated or truncated
                        )
                
                # Train agent
                loss = self.agent.train()
                
                # Soft update target network every 4 steps
                if self.current_step % 4 == 0:
                    self.agent.update_target_network()
            
            # Update observations
            observations = next_observations
            self.current_step += 1
            
            # Log rewards and survival
            for player_id in self.player_ids:
                log_player_rewards[player_id] += rewards[player_id]
                if self.alive[player_id]:
                    log_survival_times[player_id] += 1
            
            if terminated or truncated:
                break
        
        # Decay epsilon after episode
        if self.is_train:
            self.agent.decay_epsilon()
        
        # Visualization
        if visualization and visualization == "Wled":
            self.visualizer.show(All_game_actions, self.positions, self.shield_on_map)
        
        return log_player_rewards, log_survival_times, self.current_step, sum(self.alive.values())
    
    def step(self, actions: Dict[str, int]):
        """Execute one step with SINGLE action per player"""
        prev_hp = {pid: self.hp[pid] for pid in self.player_ids}
        
        rewards = {pid: 0 for pid in self.player_ids}
        All_game_actions = {}
        
        # FIXED: Execute actions in deterministic order (by player_id)
        # No priority sorting - keeps action effects predictable
        for player_id in self.player_ids:
            if not self.alive[player_id]:
                continue
            
            action_idx = actions[player_id]
            action_name = self.actions_config[action_idx]["name"]
            
            # FIXED: Reward is calculated inside _execute_action now
            reward, accepted_action = self._execute_action(player_id, action_name)
            rewards[player_id] += reward
            
            if accepted_action:
                All_game_actions[player_id] = accepted_action[0]
        
        # Regenerate mana
        for player_id in self.player_ids:
            if self.alive[player_id]:
                self.mana[player_id] = min(self.max_mana, self.mana[player_id] + self.mana_regen)
        
        # FIXED: Penalty for taking damage (NOT reward for dealing damage elsewhere)
        for player_id in self.player_ids:
            if self.alive[player_id]:
                damage_taken = prev_hp[player_id] - self.hp[player_id]
                if damage_taken > 0:
                    rewards[player_id] -= damage_taken * 0.5
        
        # Small survival reward
        for player_id in self.player_ids:
            if self.alive[player_id]:
                rewards[player_id] += 0.1
        
        # Check termination
        alive_count = sum(self.alive.values())
        terminated = alive_count <= 1
        
        # Victory bonus
        if terminated and alive_count == 1:
            winner = [pid for pid in self.player_ids if self.alive[pid]][0]
            rewards[winner] += 100
        
        # Get next observations
        next_observations = {pid: self._get_observation(pid) for pid in self.player_ids}
        truncated = self.current_step >= self.max_steps
        info = self._get_info()
        
        return next_observations, rewards, terminated, truncated, info, All_game_actions
    
    def _get_observation(self, player_id):
        """Get observation for specific player"""
        if not self.alive[player_id]:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Get alive opponents
        alive_opponents = [pid for pid in self.player_ids if pid != player_id and self.alive[pid]]
        
        # Average opponent stats
        avg_opp_hp = np.mean([self.hp[pid] for pid in alive_opponents]) if alive_opponents else 0
        avg_opp_mana = np.mean([self.mana[pid] for pid in alive_opponents]) if alive_opponents else 0
        
        # Closest opponent info
        closest_players = Game_function._get_n_closest_opponents(
            player_id=player_id,
            positions=self.positions,
            alive=self.alive,
            player_ids=self.player_ids,
            strip_length=self.strip_length,
            n=2
        )
        
        # Opponent 1
        if closest_players and closest_players[0][2] is not None:
            closest_dist_1, closest_dir_1, closest_pid_1 = closest_players[0]
            opp_1_hp = self.hp[closest_pid_1]
            opp_1_mana = self.mana[closest_pid_1]
        else:
            closest_dist_1 = self.strip_length / 2
            closest_dir_1 = 0
            opp_1_hp = 0
            opp_1_mana = 0
        
        # Opponent 2
        if len(closest_players) >= 2 and closest_players[1][2] is not None:
            closest_dist_2, closest_dir_2, closest_pid_2 = closest_players[1]
            opp_2_hp = self.hp[closest_pid_2]
            opp_2_mana = self.mana[closest_pid_2]
        else:
            closest_dist_2 = self.strip_length / 2
            closest_dir_2 = 0
            opp_2_hp = 0
            opp_2_mana = 0
        
        # Normalized observation (13 features)
        obs = np.array([
            self.hp[player_id] / self.max_hp,              # 0-1
            self.mana[player_id] / self.max_mana,          # 0-1
            avg_opp_hp / self.max_hp,                      # 0-1
            avg_opp_mana / self.max_mana,                  # 0-1
            len(alive_opponents) / (self.num_players - 1) if self.num_players > 1 else 0,  # 0-1
            closest_dist_1 / self.strip_length,            # 0-1
            closest_dir_1,                                 # -1, 0, or 1
            opp_1_hp / self.max_hp,                        # 0-1
            opp_1_mana / self.max_mana,                    # 0-1
            closest_dist_2 / self.strip_length,            # 0-1
            closest_dir_2,                                 # -1, 0, or 1
            opp_2_hp / self.max_hp,                        # 0-1
            opp_2_mana / self.max_mana,                    # 0-1
        ], dtype=np.float32)
        
        return obs
    
    def _execute_action(self, player_id, action):
        """Execute action with proper validation and reward calculation"""
        accepted_action = []
        
        if not self.alive[player_id]:
            return 0, accepted_action
        
        reward = 0
        mana_cost = self.actions_config_with_name[action].get("mana", 0)
        
        # Check mana
        if self.mana[player_id] < mana_cost:
            return -1, accepted_action  # Small penalty for invalid action
        
        self.mana[player_id] -= mana_cost
        
        # Movement
        if action == "go_left":
            self.positions[player_id] = (self.positions[player_id] - 1) % self.strip_length
            accepted_action.append({"name": action})
            reward += 0
        
        elif action == "go_right":
            self.positions[player_id] = (self.positions[player_id] + 1) % self.strip_length
            accepted_action.append({"name": action})
            reward += 0
        
        # Attacks (FIXED: Reward based on ACTUAL damage dealt)
        elif action in ["punch", "lazer"]:
            damage = self.actions_config_with_name[action]["damage"]
            attack_range = self.actions_config_with_name[action]["range"]
            
            targets_hit = []
            for target_id in self.player_ids:
                if target_id != player_id and self.alive[target_id]:
                    dist = Game_function._get_cyclic_distance(
                        self.positions[player_id],
                        self.positions[target_id],
                        self.strip_length
                    )
                    
                    if dist <= attack_range:
                        direction = Game_function._get_direction(
                            self.positions[player_id],
                            self.positions[target_id],
                            self.strip_length
                        )
                        
                        # Check shield blocking
                        shield_blocked = False
                        for i in range(1, int(dist) + 1):
                            check_pos = (self.positions[player_id] + i * direction) % self.strip_length
                            if self.shield_on_map[check_pos]:
                                shield_blocked = True
                                break
                        
                        if not shield_blocked:
                            targets_hit.append(target_id)
            
            # FIXED: Calculate damage dealt INSIDE the action
            if targets_hit:
                total_damage_dealt = 0
                for target_id in targets_hit:
                    self.hp[target_id] -= damage
                    total_damage_dealt += damage
                    
                    if self.hp[target_id] <= 0:
                        self.alive[target_id] = False
                        reward += 50  # Kill bonus
                
                # Reward based on actual damage
                reward += total_damage_dealt * 0.8
                accepted_action.append({"name": action, "hit": True})
            else:
                reward -= 2  # Miss penalty
                accepted_action.append({"name": action, "hit": False})
        
        # Shields
        elif action == "left_shield":
            shield_pos = (self.positions[player_id] - 1) % self.strip_length
            self.shield_on_map[shield_pos] = True
            accepted_action.append({"name": action})
            reward += 0
        
        elif action == "right_shield":
            shield_pos = (self.positions[player_id] + 1) % self.strip_length
            self.shield_on_map[shield_pos] = True
            accepted_action.append({"name": action})
            reward += 0
        
        # Heal
        elif action == "heal":
            heal_amount = self.actions_config_with_name[action]["regen"]
            old_hp = self.hp[player_id]
            self.hp[player_id] = min(self.max_hp, self.hp[player_id] + heal_amount)
            actual_heal = self.hp[player_id] - old_hp
            reward += actual_heal * 0.3
            accepted_action.append({"name": action})
        
        # Teleport
        elif action == "random_tp":
            new_pos = random.randint(0, self.strip_length - 1)
            self.positions[player_id] = new_pos
            accepted_action.append({"name": action, "new_pos": new_pos})
            reward += 0
        
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