import numpy as np
from Function.core.game_env import Combat1DEnv
from Function.agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from collections import deque
import os

def train_self_play(num_episodes=5000, num_players=2, max_steps_per_episode=900, 
                    save_interval=500, visualize_interval=100):
    """
    Train a single DQN agent through self-play
    The same model controls all players
    """
    
    # Create environment
    env = Combat1DEnv(strip_length=300, num_players=num_players)
    
    # Create ONE agent for self-play
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    survival_times = {pid: deque(maxlen=100) for pid in env.player_ids}
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_progress", exist_ok=True)
    
    print("=" * 70)
    print(f"Starting Self-Play Training with {num_players} players")
    print(f"Device: {agent.device}")
    print("=" * 70)
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        observations, info = env.reset()
        
        # Track episode data
        player_rewards = {pid: 0 for pid in env.player_ids}
        player_steps = {pid: 0 for pid in env.player_ids}
        done = False
        
        total_steps = 0
        while not done and total_steps < max_steps_per_episode:
            # All players use the SAME agent
            actions = {}
            for player_id in env.player_ids:
                if env.alive[player_id]:
                    obs = observations[player_id]
                    actions[player_id] = agent.select_action(obs, training=True)
            
            # Step environment
            next_observations, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # Store transitions and train for each player
            for player_id in env.player_ids:
                if player_id in actions:
                    agent.store_transition(
                        observations[player_id],
                        actions[player_id],
                        rewards[player_id],
                        next_observations[player_id],
                        not env.alive[player_id] or done
                    )
                    
                    # Train after each step
                    agent.train()
                    
                    player_rewards[player_id] += rewards[player_id]
                    if env.alive[player_id]:
                        player_steps[player_id] += 1
            
            observations = next_observations
            total_steps += 1

            # end episode if no player are moving because of layzyness
            liste = []
            for pid in env.player_ids:
                liste.append(env.not_moving_step[pid])
            if min(liste) == 15:
                done = True
        
        # Record statistics
        avg_reward = np.mean(list(player_rewards.values()))
        episode_rewards.append(avg_reward)
        episode_lengths.append(total_steps)
        
        # Record survival times
        for player_id in env.player_ids:
            survival_times[player_id].append(player_steps[player_id])
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Print progress
        if episode % 10 == 0:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_survival = {
                pid: np.mean(survival_times[pid]) 
                for pid in env.player_ids
            }
            
            print(f"Ep {episode:4d} | "
                  f"Steps: {total_steps:3d} | "
                  f"Avg Reward: {avg_reward_100:7.2f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Survivors: {info['num_alive']} | "
                  f"ε: {agent.epsilon:.3f}")
            
            if episode % 50 == 0:
                print(f"  Avg Survival: {', '.join([f'{pid}: {avg_survival[pid]:.0f}' for pid in env.player_ids])}")
        
        # Visualize progress
        if episode % visualize_interval == 0:
            plot_training_progress(episode_rewards, episode_lengths, survival_times, episode)
        
        # Save model
        if episode % save_interval == 0:
            agent.save(f"models/agent_self_play_ep{episode}.pth")
            print(f"Model saved at episode {episode}")
    
    # Final save
    agent.save("models/agent_self_play_final.pth")
    print("\nTraining completed!")
    
    return agent, episode_rewards


def plot_training_progress(rewards, lengths, survival_times, episode):
    """Plot training statistics"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    window = 100
    
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


if __name__ == "__main__":
    # Train with 2 players (classic duel)
    train_self_play(num_episodes=5000, num_players=2)
    
    # Or train with 4 players (battle royale)
    # train_self_play(num_episodes=10000, num_players=4)