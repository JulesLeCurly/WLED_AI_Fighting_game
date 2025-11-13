import numpy as np
import matplotlib.pyplot as plt
import os


def plot_training_progress(rewards, lengths, survival_times, episode):
    """Plot training statistics"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    window = 100

    os.makedirs('training_progress', exist_ok=True)
    
    
    # Plot 1: Average Reward
    if len(rewards) >= window:
        avg_rewards = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        axes[0].plot(avg_rewards, color='purple', linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Average Reward')
        axes[0].set_title(f'Training Progress - Episode {episode}')
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Episode Length
    if len(lengths) >= window:
        avg_lengths = [np.mean(lengths[max(0, i-window):i+1]) for i in range(len(lengths))]
        axes[1].plot(avg_lengths, color='orange', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Episode Length (steps)')
        axes[1].set_title('Combat Duration')
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Survival Times per Player
    colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan']
    for i, (player_id, times) in enumerate(survival_times.items()):
        if len(times) >= window:
            survival_avg = [np.mean(list(times)[max(0, j-window):j+1]) for j in range(len(times))]
            axes[2].plot(survival_avg, label=player_id, color=colors[i % len(colors)], 
                        linewidth=2, alpha=0.8)
    
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Average Survival Time (steps)')
    axes[2].set_title('Player Performance Balance')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_progress/training_self_play_ep{episode}.png', dpi=100)
    plt.close()