import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple
import random

class Combat1DEnv(gym.Env):
    """
    1D Combat Environment with N-player support
    Players identified by unique IDs in dictionaries
    """
    
    def __init__(self, strip_length=300, num_players=2):
        super().__init__()
        
        # World parameters
        self.strip_length = strip_length
        self.num_players = num_players
        self.max_steps = 900  # 30 seconds at 30 FPS
        
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
        self.max_hp = 100
        self.max_mana = 100
        self.mana_regen = 2
        self.move_speed = 2
        
        # Action configuration
        self.actions_config = {
            0: {"name": "left", "mana": 5, "range": 0, "damage": 0},
            1: {"name": "right", "mana": 5, "range": 0, "damage": 0},
            2: {"name": "attack_short", "mana": 10, "range": 10, "damage": 15},
            3: {"name": "attack_long", "mana": 25, "range": 50, "damage": 10},
            4: {"name": "rest", "mana": 0, "range": 0, "damage": 0}
        }
        
        # Spaces
        self.action_space = spaces.Discrete(len(self.actions_config))

        self.Observation_dict = {
            "my_hp": {"low": 0, "high": self.max_hp},
            "my_mana": {"low": 0, "high": self.max_mana},
            "avg_opp_hp": {"low": 0, "high": self.max_hp},
            "avg_opp_mana": {"low": 0, "high": self.max_mana},
            "num_alive": {"low": 0, "high": self.num_players},

            "closest_distance_to_opp_1": {"low": 0, "high": self.strip_length},
            "closest_direction_to_opp_1": {"low": -1, "high": 1},
            "opp_1_hp": {"low": 0, "high": self.max_hp},
            "opp_1_mana": {"low": 0, "high": self.max_mana},

            "closest_distance_to_opp_2": {"low": 0, "high": self.strip_length},
            "closest_direction_to_opp_2": {"low": -1, "high": 1},
            "opp_2_hp": {"low": 0, "high": self.max_hp},
            "opp_2_mana": {"low": 0, "high": self.max_mana},
        }
        
        self.observation_space = spaces.Box(
            low=np.array([dict_observation_one_ellement["low"] for dict_observation_one_ellement in self.Observation_dict.values()]),
            high=np.array([dict_observation_one_ellement["high"] for dict_observation_one_ellement in self.Observation_dict.values()]),
            dtype=np.float32
        )
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        players_shuffle = self.player_ids.copy()
        random.shuffle(players_shuffle)

        # Initialize player positions (evenly distributed)
        spacing = self.strip_length / self.num_players
        self.positions = {
            pid: spacing * i for i, pid in enumerate(players_shuffle)
        }
        
        # Initialize stats
        self.hp = {pid: self.max_hp for pid in self.player_ids}
        self.mana = {pid: self.max_mana for pid in self.player_ids}
        self.alive = {pid: True for pid in self.player_ids}
        self.not_moving_step = {pid: 0 for pid in self.player_ids}
        
        # Last actions for visualization
        self.last_actions = {pid: None for pid in self.player_ids}
        
        # Attack effects (for visualization)
        self.active_attacks = []  # List of (pos, direction, range, color)
        
        # Step counter
        self.current_step = 0
        
        # Get observations for all players
        observations = {pid: self._get_observation(pid) for pid in self.player_ids}
        info = self._get_info()
        
        return observations, info
    
    def _get_cyclic_distance(self, pos1, pos2):
        """Calculate shortest distance on cyclic world"""
        direct = abs(pos2 - pos1)
        wrap = self.strip_length - direct
        return min(direct, wrap)
    
    def _get_direction(self, from_pos, to_pos):
        """Get direction: -1 (left), +1 (right)"""
        direct = to_pos - from_pos
        wrap_right = (to_pos + self.strip_length) - from_pos
        wrap_left = to_pos - (from_pos + self.strip_length)
        
        min_dist = min(abs(direct), abs(wrap_right), abs(wrap_left))
        
        if abs(direct) == min_dist:
            return 1 if direct > 0 else -1
        elif abs(wrap_right) == min_dist:
            return 1
        else:
            return -1
    
    def _get_closest_opponent(self, player_id):
        """Find closest alive opponent"""
        my_pos = self.positions[player_id]
        min_dist = float('inf')
        closest_pos = None
        
        for pid in self.player_ids:
            if pid != player_id and self.alive[pid]:
                dist = self._get_cyclic_distance(my_pos, self.positions[pid])
                if dist < min_dist:
                    min_dist = dist
                    closest_pos = self.positions[pid]
        
        if closest_pos is None:
            return self.strip_length / 2, 0  # No opponents
        
        direction = self._get_direction(my_pos, closest_pos)
        return min_dist, direction
    
    def _get_n_closest_opponents(self, player_id, n=3):
        """Retourne les n adversaires vivants les plus proches d'un joueur donné."""
        my_pos = self.positions[player_id]
        distances = []

        # Calculer la distance à chaque adversaire vivant
        for pid in self.player_ids:
            if pid != player_id and self.alive[pid]:
                dist = self._get_cyclic_distance(my_pos, self.positions[pid])
                direction = self._get_direction(my_pos, self.positions[pid])
                distances.append((dist, direction, pid))

        # Trier les adversaires par distance croissante
        distances.sort(key=lambda x: x[0])

        # Garder seulement les n plus proches
        n_closest = distances[:n]

        # Si aucun adversaire vivant
        if not n_closest:
            return [(self.strip_length / 2, 0, None)]

        return n_closest

    
    def _get_observation(self, player_id):
        """Get observation for a specific player"""
        if not self.alive[player_id]:
            return np.zeros(len(self.Observation_dict), dtype=np.float32)
        
        # Calculate average opponent HP
        alive_opponents = [pid for pid in self.player_ids if pid != player_id and self.alive[pid]]
        
        avg_opp_hp = np.mean([self.hp[pid] for pid in alive_opponents]) if alive_opponents else 0
        avg_opp_mana = np.mean([self.mana[pid] for pid in alive_opponents]) if alive_opponents else 0

        closest_players = self._get_n_closest_opponents(player_id=player_id, n=2)

        # Find closest opponent
        closest_dist_opp_1, closest_dir_opp_1, closest_pid_opp_1 = closest_players[0][0], closest_players[0][1], closest_players[0][2]
        closest_dist_opp_2, closest_dir_opp_2, closest_pid_opp_2 = closest_players[0][0], closest_players[0][1], closest_players[0][2]

        opp_1_hp = self.hp[closest_pid_opp_1] if closest_pid_opp_1 is not None else 0
        opp_2_hp = self.hp[closest_pid_opp_2] if closest_pid_opp_2 is not None else 0

        opp_1_mana = self.mana[closest_pid_opp_1] if closest_pid_opp_1 is not None else 0
        opp_2_mana = self.mana[closest_pid_opp_2] if closest_pid_opp_2 is not None else 0

        
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
        ])
    
    def _execute_action(self, player_id, action):
        """Execute action for a player and return reward"""
        if not self.alive[player_id]:
            return 0
        
        reward = -0.01  # Small time penalty
        
        pos = self.positions[player_id]
        mana = self.mana[player_id]
        action_config = self.actions_config[action]
        
        # Check mana
        if mana < action_config["mana"]:
            return reward - 5  # Invalid action penalty
        
        # Execute movement
        if action == 0:  # Left
            pos = (pos - self.move_speed) % self.strip_length
        elif action == 1:  # Right
            pos = (pos + self.move_speed) % self.strip_length
        
        # Execute attacks
        elif action in [2, 3]:
            damage = action_config["damage"]
            attack_range = action_config["range"]
            
            # Find all targets in range
            targets_hit = []
            for target_id in self.player_ids:
                if target_id != player_id and self.alive[target_id]:
                    dist = self._get_cyclic_distance(pos, self.positions[target_id])
                    if dist <= attack_range:
                        targets_hit.append(target_id)
            
            if targets_hit:
                # Hit someone!
                for target_id in targets_hit:
                    self.hp[target_id] -= damage
                    if self.hp[target_id] <= 0:
                        self.alive[target_id] = False
                        reward += 50  # Kill bonus
                reward += 10 * len(targets_hit)  # Hit reward
                
                # Store attack for visualization
                direction = self._get_direction(pos, self.positions[targets_hit[0]])
                self.active_attacks.append({
                    'pos': pos,
                    'direction': direction,
                    'range': attack_range,
                    'color': self.player_colors[player_id],
                    'steps_left': 2  # Display for 2 steps
                })
            elif action == 4:
                # Don't move
                self.not_moving_step[player_id] += 1
                if self.not_moving_step[player_id] >= 10:
                    reward -= 10
            else:
                # Miss
                reward -= 5
        
        # Update state
        self.positions[player_id] = pos
        self.mana[player_id] = mana - action_config["mana"]
        self.last_actions[player_id] = action
        
        return reward
    
    def step(self, actions: Dict[str, int]):
        """
        Execute one step with all players acting simultaneously
        actions: dict {player_id: action}
        """
        # Store initial HP for damage calculation
        prev_hp = {pid: self.hp[pid] for pid in self.player_ids}
        
        # Execute all actions
        rewards = {}
        for player_id in self.player_ids:
            if player_id in actions and self.alive[player_id]:
                rewards[player_id] = self._execute_action(player_id, actions[player_id])
            else:
                rewards[player_id] = 0
        
        # Regenerate mana for alive players
        for player_id in self.player_ids:
            if self.alive[player_id]:
                self.mana[player_id] = min(self.max_mana, self.mana[player_id] + self.mana_regen)
        
        # Add damage-based rewards
        for player_id in self.player_ids:
            if self.alive[player_id]:
                damage_dealt = prev_hp[player_id] - self.hp[player_id]
                rewards[player_id] += damage_dealt * 1.0
        
        # Subtract rewards for player that are on each other
        for player_id in self.player_ids:
            if self.alive[player_id]:
                for target_id in self.player_ids:
                    if target_id != player_id and self.alive[target_id]:
                        dist = self._get_cyclic_distance(self.positions[player_id], self.positions[target_id])
                        if dist == 0:
                            rewards[player_id] -= 10

        
        # Update attack effects
        self.active_attacks = [
            {**atk, 'steps_left': atk['steps_left'] - 1}
            for atk in self.active_attacks if atk['steps_left'] > 0
        ]
        
        self.current_step += 1
        
        # Check game end
        alive_count = sum(self.alive.values())
        terminated = alive_count <= 1 or self.current_step >= self.max_steps
        
        if terminated:
            # Victory bonus for survivors
            for player_id in self.player_ids:
                if self.alive[player_id]:
                    rewards[player_id] += 100
        
        # Get observations
        observations = {pid: self._get_observation(pid) for pid in self.player_ids}
        truncated = self.current_step >= self.max_steps
        info = self._get_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _get_info(self):
        """Return game state info"""
        return {
            'positions': self.positions.copy(),
            'hp': self.hp.copy(),
            'mana': self.mana.copy(),
            'alive': self.alive.copy(),
            'last_actions': self.last_actions.copy(),
            'step': self.current_step,
            'num_alive': sum(self.alive.values())
        }
    
    def render(self):
        """Return LED strip state for visualization"""
        strip = np.zeros((self.strip_length, 3))
        
        # Draw active attacks first (background layer)
        for attack in self.active_attacks:
            pos = int(attack['pos'])
            direction = attack['direction']
            for i in range(1, attack['range'] + 1):
                idx = (pos + i * direction) % self.strip_length
                # Fade effect
                intensity = 1.0 - (i / attack['range']) * 0.5
                color = np.array(attack['color']) * intensity * (attack['steps_left'] / 2)
                strip[idx] = np.clip(strip[idx] + color, 0, 255)
        
        # Draw players (foreground layer)
        for player_id in self.player_ids:
            if self.alive[player_id]:
                idx = int(self.positions[player_id]) % self.strip_length
                strip[idx] = self.player_colors[player_id]
        
        return strip.astype(np.uint8)