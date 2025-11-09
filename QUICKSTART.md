# Quick Start Guide

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd WLED_AI_Fighting_game
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Test the environment
```bash
python Main.py test
```

This will verify that everything is installed correctly and run a quick simulation.

### Train the agents
```bash
# Quick training (5000 episodes, ~30-60 min)
python Main.py train

# Extended training (10000 episodes for better results)
python Main.py train --episodes 10000
```

**What happens during training:**
- Models are saved every 500 episodes in `models/` folder
- Training plots are generated every 100 episodes
- Progress is printed every 10 episodes
- Look for win rates approaching 50% (balanced agents)

### Visualize trained agents
```bash
# Watch 3 games
python Main.py visualize

# Watch 10 games
python Main.py visualize --games 10

# Use specific models
python Main.py visualize --agent1 models/agent1_episode_2500.pth --agent2 models/agent2_episode_2500.pth
```

## Expected Training Progress

### Episodes 0-500 (Random exploration)
- Win rate: Highly variable (60-40 or 40-60)
- Behavior: Random movement, attacking at wrong distances
- Epsilon: 1.0 → 0.6

### Episodes 500-1500 (Learning basics)
- Win rate: Starting to balance (55-45)
- Behavior: Learning to move toward opponent
- Epsilon: 0.6 → 0.2

### Episodes 1500-3000 (Tactical learning)
- Win rate: Approaching 50-50
- Behavior: Understanding attack ranges, basic mana management
- Epsilon: 0.2 → 0.05

### Episodes 3000-5000 (Mastery)
- Win rate: Stable around 50-50
- Behavior: Strategic positioning, optimal attack timing
- Epsilon: 0.05 → 0.01

## File Structure

```
WLED_AI_Fighting_game/
├── Main.py                  # Entry point
├── README.md               # Full documentation
├── QUICKSTART.md           # This file
├── requirements.txt        # Dependencies
├── Function/
│   ├── __init__.py        # Package init
│   ├── game_env.py        # Game environment
│   ├── dqn_agent.py       # DQN agent
│   ├── train.py           # Training script
│   └── visualizer.py      # Visualization
└── models/                # Saved models (created during training)
    ├── agent1_episode_500.pth
    ├── agent2_episode_500.pth
    └── ...
```

## Troubleshooting

### Import errors
Make sure you run scripts from the project root directory:
```bash
cd WLED_AI_Fighting_game
python Main.py train
```

### CUDA/GPU issues
PyTorch will automatically use GPU if available. To force CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
python Main.py train
```

### One agent dominates
This is normal in early training. Continue training - agents will balance out through self-play.

### Slow training
- Use GPU if available (20x faster)
- Reduce episodes for quick test: `--episodes 1000`
- Training is CPU-intensive; close other applications

## Next Steps

After training:
1. Watch the visualizer to see strategies
2. Modify game rules in `Function/game_env.py`
3. Add new actions (dash, shield, etc.)
4. Integrate with WLED hardware

## Need Help?

Check the full [README.md](README.md) for:
- Detailed code explanations
- Customization guide
- WLED integration instructions
- Training tips and tricks