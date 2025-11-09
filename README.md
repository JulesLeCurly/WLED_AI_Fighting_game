# 1D Combat Arena - Reinforcement Learning Battle

Two AI agents learn to fight each other in a 1D cyclic world, designed for WLED LED strip visualization.

## Installation

### Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

### Install dependencies

```bash
pip install torch gymnasium numpy matplotlib
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train the agents

```bash
python train.py
```

This will:
- Train two DQN agents for 5000 episodes
- Save models every 500 episodes in `models/`
- Generate training progress plots every 100 episodes
- Print statistics every 10 episodes

Training takes approximately 30-60 minutes on CPU (faster with GPU).

### 2. Visualize trained agents

```bash
python visualizer.py
```

This will load the trained agents and display their combat in real-time with:
- LED strip visualization (300 pixels)
- HP and Mana bars
- Action indicators
- HP history graph

## Game Rules

### World
- **300 pixels** LED strip (cyclic/wraparound world)
- **30 FPS** game speed
- **Max 30 seconds** per match (900 steps)

### Player Stats
- **HP**: 100 (health points)
- **Mana**: 100 (regenerates +2 per step)
- **Move speed**: 2 pixels per step

### Actions

| Action | Mana Cost | Range | Damage | Effect |
|--------|-----------|-------|--------|--------|
| **Move Left** | 0 | - | - | Move 2 pixels left |
| **Move Right** | 0 | - | - | Move 2 pixels right |
| **Short Attack** | 10 | 10px | 15 HP | Quick melee attack |
| **Long Attack** | 25 | 50px | 10 HP | Ranged projectile |

### Rewards (Reinforcement Learning)

- **+100**: Win the match
- **+10**: Successfully hit opponent
- **-10**: Get hit by opponent
- **-5**: Miss an attack (waste mana)
- **-0.01**: Each step (encourages finishing quickly)

## Understanding the Code

### game_env.py - The Environment

The core game logic follows the Gymnasium API:

```python
env = Combat1DEnv(strip_length=300)
(obs_p1, obs_p2), info = env.reset()

action_p1 = 1  # Move right
action_p2 = 2  # Short attack

(obs_p1, obs_p2), (reward_p1, reward_p2), done, truncated, info = env.step((action_p1, action_p2))
```

**Observation space** (6 values per agent):
- Distance to opponent (0-150)
- Direction to opponent (-1 = left, +1 = right)
- My HP (0-100)
- My Mana (0-100)
- Opponent HP (0-100)
- Opponent Mana (0-100)

### dqn_agent.py - The AI Brain

Implements Deep Q-Network with:
- **Neural network**: 128→128→64 hidden layers
- **Experience replay**: 10,000 transition buffer
- **Epsilon-greedy exploration**: starts at 1.0, decays to 0.01
- **Target network**: updated every 10 episodes

```python
agent = DQNAgent(state_size=6, action_size=4)
action = agent.select_action(observation, training=True)
agent.store_transition(state, action, reward, next_state, done)
agent.train()  # Learn from experience replay
```

### train.py - Training Loop

Self-play training where both agents improve together:

```python
for episode in range(5000):
    # Both agents play simultaneously
    action_p1 = agent1.select_action(obs_p1)
    action_p2 = agent2.select_action(obs_p2)
    
    # Environment updates
    (obs_p1, obs_p2), (reward_p1, reward_p2), done = env.step((action_p1, action_p2))
    
    # Both agents learn
    agent1.train()
    agent2.train()
```

## Customization

### Modify game parameters

Edit `game_env.py`:

```python
# Change world size
env = Combat1DEnv(strip_length=500)

# Adjust stats in __init__
self.max_hp = 150
self.mana_regen = 3
self.move_speed = 3
```

### Add new actions

In `game_env.py`, add to `actions_config`:

```python
self.actions_config = {
    # ... existing actions ...
    4: {"name": "dash", "mana": 20, "range": 0, "damage": 0},  # New action
}
```

Then update `action_space`:

```python
self.action_space = spaces.Discrete(5)  # Now 5 actions
```

### Adjust training hyperparameters

In `train.py` or `dqn_agent.py`:

```python
agent = DQNAgent(
    state_size=6, 
    action_size=4,
    learning_rate=0.0005,  # Lower = more stable, slower
    gamma=0.95             # Discount factor for future rewards
)

# In train.py
train_agents(
    num_episodes=10000,    # More episodes = better learning
    save_interval=1000,
    visualize_interval=200
)
```

## WLED Integration (Future)

To connect to WLED, you'll need to:

1. Get the LED strip state from `env.render()` (returns 300×3 RGB array)
2. Send via HTTP to WLED's JSON API:

```python
import requests

strip_data = env.render()  # Shape: (300, 3)
led_array = strip_data.flatten().tolist()  # Convert to flat list

requests.post(
    "http://YOUR_WLED_IP/json/state",
    json={"seg": {"i": led_array}}
)
```

See WLED documentation for real-time UDP streaming (faster than HTTP).

## Training Tips

1. **Monitor win rates**: Balanced agents should have ~50% win rate each
2. **Watch epsilon decay**: Agents explore less over time (printed during training)
3. **Check rewards**: Should increase over episodes as agents learn
4. **Visualize early**: Run `visualizer.py` after 1000 episodes to see progress

## Troubleshooting

**Agents don't learn (rewards stay flat)**
- Increase training episodes (10,000+)
- Adjust learning rate (try 0.0005 or 0.002)
- Check reward shaping in `game_env.py`

**One agent dominates**
- This is normal early in training
- Continue training - they should balance out
- Try training against different opponents

**CUDA out of memory**
- Reduce batch size in `dqn_agent.py`: `self.batch_size = 32`
- Use CPU: PyTorch will automatically fallback

## Next Steps

- [ ] Add dash/teleport action
- [ ] Implement directional shields
- [ ] Add bonus pickups on map
- [ ] Create spell system with cooldowns
- [ ] Integrate with WLED hardware
- [ ] Add 14-LED info display for HP bars
- [ ] Implement tournament mode (multiple agents)

## License

MIT License - Feel free to use and modify for your projects!

## Credits

Created for WLED LED strip visualization using Deep Reinforcement Learning.