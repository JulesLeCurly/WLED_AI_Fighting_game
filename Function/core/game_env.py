import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Combat1DEnv(gym.Env):
    """
    1D Combat Environment for Reinforcement Learning
    Two agents fight on a cyclic 1D world (300 pixels LED strip)
    """
    
    def __init__(self, strip_length=300):
        super().__init__()
        
        # World parameters
        self.strip_length = strip_length
        self.max_steps = 900  # 30 seconds at 30 FPS
        
        # Game parameters
        self.max_hp = 100
        self.max_mana = 100
        self.mana_regen = 2
        self.move_speed = 2
        
        # Action costs and effects
        self.actions_config = {
            0: {"name": "left", "mana": 5, "range": 0, "damage": 0},
            1: {"name": "right", "mana": 5, "range": 0, "damage": 0},
            2: {"name": "attack_short", "mana": 10, "range": 3, "damage": 15},
            3: {"name": "attack_long", "mana": 25, "range": 10, "damage": 10},
            4: {"name": "rest", "mana": 0, "range": 0, "damage": 0}
        }
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(len(self.actions_config))
        
        # Observation: [distance, direction, my_hp, my_mana, opp_hp, opp_mana]
        self.observation_space = spaces.Box(
            low=np.array([0, -1, 0, 0, 0, 0]),
            high=np.array([150, 1, 100, 100, 100, 100]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize player positions
        self.p1_pos = 75.0
        self.p2_pos = 225.0
        
        # Initialize stats
        self.p1_hp = self.max_hp
        self.p2_hp = self.max_hp
        self.p1_mana = self.max_mana
        self.p2_mana = self.max_mana
        
        # Step counter
        self.current_step = 0
        
        # Last actions for visualization
        self.p1_last_action = None
        self.p2_last_action = None
        
        # Get initial observations
        obs_p1 = self._get_observation(player=1)
        obs_p2 = self._get_observation(player=2)
        
        info = self._get_info()
        
        return (obs_p1, obs_p2), info
    
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
        
        distances = {
            direct: 0,
            wrap_right: 1,
            wrap_left: -1
        }
        
        min_dist = min(abs(direct), abs(wrap_right), abs(wrap_left))
        
        for dist, direction in distances.items():
            if abs(dist) == min_dist:
                return 1 if dist > 0 else -1
        
        return 0
    
    def _get_observation(self, player):
        """Get observation for a specific player"""
        if player == 1:
            my_pos, opp_pos = self.p1_pos, self.p2_pos
            my_hp, opp_hp = self.p1_hp, self.p2_hp
            my_mana, opp_mana = self.p1_mana, self.p2_mana
        else:
            my_pos, opp_pos = self.p2_pos, self.p1_pos
            my_hp, opp_hp = self.p2_hp, self.p1_hp
            my_mana, opp_mana = self.p2_mana, self.p1_mana
        
        distance = self._get_cyclic_distance(my_pos, opp_pos)
        direction = self._get_direction(my_pos, opp_pos)
        
        return np.array([
            distance / self.strip_length,
            direction,
            my_hp / self.max_hp,
            my_mana / self.max_mana,
            opp_hp / self.max_hp,
            opp_mana / self.max_mana
        ], dtype=np.float32)
        
    
    def _execute_action(self, player, action):
        """Execute action for a player and return reward"""
        reward = -0.01  # Small penalty for each step
        
        if player == 1:
            pos, mana = self.p1_pos, self.p1_mana
            opp_pos = self.p2_pos
        else:
            pos, mana = self.p2_pos, self.p2_mana
            opp_pos = self.p1_pos
        
        action_config = self.actions_config[action]
        
        # Check if enough mana
        if mana < action_config["mana"]:
            return reward - 5  # Penalty for invalid action
        
        # Execute action
        if action == 0 and mana >= 5:  # Left
            pos = (pos - self.move_speed) % self.strip_length
        elif action == 1 and mana >= 5:  # Right
            pos = (pos + self.move_speed) % self.strip_length
        elif action in [2, 3]:  # Attacks
            distance = self._get_cyclic_distance(pos, opp_pos)
            
            if distance <= action_config["range"]:
                # Hit!
                damage = action_config["damage"]
                if player == 1:
                    self.p2_hp -= damage
                else:
                    self.p1_hp -= damage
                reward += 10  # Reward for hitting
            else:
                # Miss
                reward -= 5  # Penalty for missing
            
        mana -= action_config["mana"]
        
        # Update player state
        if player == 1:
            self.p1_pos = pos
            self.p1_mana = mana
            self.p1_last_action = action
        else:
            self.p2_pos = pos
            self.p2_mana = mana
            self.p2_last_action = action
        
        return reward
    
    def step(self, actions):
        """
        Execute one step with both players acting simultaneously
        actions: tuple (action_p1, action_p2)
        """
        action_p1, action_p2 = actions
        
        # Execute actions
        prev_p1_hp, prev_p2_hp = self.p1_hp, self.p2_hp

        reward_p1 = self._execute_action(1, action_p1)
        reward_p2 = self._execute_action(2, action_p2)
        
        # Regenerate mana
        self.p1_mana = min(self.max_mana, self.p1_mana + self.mana_regen)
        self.p2_mana = min(self.max_mana, self.p2_mana + self.mana_regen)
        
        # Check damage rewards
        damage_done_p1 = prev_p2_hp - self.p2_hp
        damage_done_p2 = prev_p1_hp - self.p1_hp

        reward_p1 += damage_done_p1 * 1.0  # proportionnel au dégât infligé
        reward_p2 += damage_done_p2 * 1.0

        
        self.current_step += 1
        
        # Check termination
        terminated = False
        if self.p1_hp <= 0:
            reward_p1 -= 100
            reward_p2 += 100
            terminated = True
        elif self.p2_hp <= 0:
            reward_p1 += 100
            reward_p2 -= 100
            terminated = True
        elif self.current_step >= self.max_steps:
            # Timeout: winner is player with most HP
            if self.p1_hp > self.p2_hp:
                reward_p1 += 50
                reward_p2 -= 50
            elif self.p2_hp > self.p1_hp:
                reward_p2 += 50
                reward_p1 -= 50
            terminated = True
        
        # Get observations
        obs_p1 = self._get_observation(player=1)
        obs_p2 = self._get_observation(player=2)
        
        truncated = self.current_step >= self.max_steps
        info = self._get_info()
        
        return (obs_p1, obs_p2), (reward_p1, reward_p2), terminated, truncated, info
    
    def _get_info(self):
        """Return additional info for debugging/visualization"""
        return {
            "p1_pos": self.p1_pos,
            "p2_pos": self.p2_pos,
            "p1_hp": self.p1_hp,
            "p2_hp": self.p2_hp,
            "p1_mana": self.p1_mana,
            "p2_mana": self.p2_mana,
            "p1_action": self.p1_last_action,
            "p2_action": self.p2_last_action,
            "step": self.current_step
        }
    
    def render(self):
        """Return LED strip state for visualization"""
        strip = np.zeros((self.strip_length, 3))
        
        # Player 1 (Red)
        p1_idx = int(self.p1_pos) % self.strip_length
        strip[p1_idx] = [255, 0, 0]
        
        # Player 2 (Blue)
        p2_idx = int(self.p2_pos) % self.strip_length
        strip[p2_idx] = [0, 0, 255]
        
        # Attack visualizations
        if self.p1_last_action == 2:  # Short attack
            for i in range(1, 11):
                idx = (p1_idx + i) % self.strip_length
                strip[idx] = [255, 255, 255]
        elif self.p1_last_action == 3:  # Long attack
            direction = self._get_direction(self.p1_pos, self.p2_pos)
            for i in range(1, 51):
                idx = (p1_idx + i * direction) % self.strip_length
                strip[idx] = [255, 255, 0]
        
        if self.p2_last_action == 2:
            for i in range(1, 11):
                idx = (p2_idx + i) % self.strip_length
                strip[idx] = [255, 255, 255]
        elif self.p2_last_action == 3:
            direction = self._get_direction(self.p2_pos, self.p1_pos)
            for i in range(1, 51):
                idx = (p2_idx + i * direction) % self.strip_length
                strip[idx] = [255, 255, 0]
        
        return strip