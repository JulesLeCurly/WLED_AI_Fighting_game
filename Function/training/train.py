import numpy as np
from Function.core.game_env import Combat1DEnv
from Function.agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from collections import deque
import os

def train_agents(num_episodes=5000, max_steps_per_episode=300, save_interval=500, visualize_interval=100):
    """Train two DQN agents to fight each other"""
    
    # Create environment
    env = Combat1DEnv(strip_length=300)
    
    # Create agents
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)
    
    # Training statistics
    episode_rewards_p1 = []
    episode_rewards_p2 = []
    p1_wins = deque(maxlen=100)
    p2_wins = deque(maxlen=100)
    
    # Create save directory
    os.makedirs("models", exist_ok=True)
    
    print("Starting training...")
    print(f"Device: {agent1.device}")
    print("-" * 60)
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        (obs_p1, obs_p2), info = env.reset()
        
        total_reward_p1 = 0
        total_reward_p2 = 0
        done = False
        
        Total_steps = 0
        while not done:
            # Select actions
            action_p1 = agent1.select_action(obs_p1, training=True)
            action_p2 = agent2.select_action(obs_p2, training=True)
            
            # Step environment
            (next_obs_p1, next_obs_p2), (reward_p1, reward_p2), terminated, truncated, info = env.step((action_p1, action_p2))
            done = terminated or truncated
            
            # Store transitions
            agent1.store_transition(obs_p1, action_p1, reward_p1, next_obs_p1, done)
            agent2.store_transition(obs_p2, action_p2, reward_p2, next_obs_p2, done)
            
            # Train agents
            agent1.train()
            agent2.train()
            
            # Update observations
            obs_p1 = next_obs_p1
            obs_p2 = next_obs_p2
            
            total_reward_p1 += reward_p1
            total_reward_p2 += reward_p2

            if Total_steps >= max_steps_per_episode:
                break
            Total_steps += 1
        
        # Record statistics
        episode_rewards_p1.append(total_reward_p1)
        episode_rewards_p2.append(total_reward_p2)
        
        # Track wins
        if info['p1_hp'] > info['p2_hp']:
            p1_wins.append(1)
            p2_wins.append(0)
        elif info['p2_hp'] > info['p1_hp']:
            p1_wins.append(0)
            p2_wins.append(1)
        else:
            p1_wins.append(0.5)
            p2_wins.append(0.5)
        
        # Decay epsilon
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        # Update target networks
        if episode % 10 == 0:
            agent1.update_target_network()
            agent2.update_target_network()
        
        # Print progress
        if episode % 10 == 0:
            avg_reward_p1 = np.mean(episode_rewards_p1[-100:])
            avg_reward_p2 = np.mean(episode_rewards_p2[-100:])
            p1_winrate = np.mean(p1_wins) * 100
            p2_winrate = np.mean(p2_wins) * 100
            
            print(f"Episode {episode:4d} | "
                  f"Total Steps: {Total_steps} | "
                  f"P1 Reward: {avg_reward_p1:7.2f} | "
                  f"P2 Reward: {avg_reward_p2:7.2f} | "
                  f"P1 WR: {p1_winrate:5.1f}% | "
                  f"P2 WR: {p2_winrate:5.1f}% | "
                  f"ε: {agent1.epsilon:.3f}")
        
        # Visualize progress
        if episode % visualize_interval == 0:
            plot_training_progress(episode_rewards_p1, episode_rewards_p2, p1_wins, p2_wins, episode)
        
        # Save models
        if episode % save_interval == 0:
            agent1.save(f"models/agent1_episode_{episode}.pth")
            agent2.save(f"models/agent2_episode_{episode}.pth")
            print(f"Models saved at episode {episode}")
    
    # Final save
    agent1.save("models/agent1_final.pth")
    agent2.save("models/agent2_final.pth")
    print("\nTraining completed!")
    
    return agent1, agent2, episode_rewards_p1, episode_rewards_p2


def plot_training_progress(rewards_p1, rewards_p2, wins_p1, wins_p2, episode):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    window = 100
    if len(rewards_p1) >= window:
        avg_rewards_p1 = [np.mean(rewards_p1[max(0, i-window):i+1]) for i in range(len(rewards_p1))]
        avg_rewards_p2 = [np.mean(rewards_p2[max(0, i-window):i+1]) for i in range(len(rewards_p2))]
        
        axes[0].plot(avg_rewards_p1, label='Agent 1 (Red)', color='red', alpha=0.8)
        axes[0].plot(avg_rewards_p2, label='Agent 2 (Blue)', color='blue', alpha=0.8)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Average Reward')
        axes[0].set_title(f'Training Progress (Episode {episode})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot win rates
    if len(wins_p1) >= window:
        win_rates_p1 = [np.mean(list(wins_p1)[max(0, i-window):i+1]) * 100 for i in range(len(wins_p1))]
        win_rates_p2 = [np.mean(list(wins_p2)[max(0, i-window):i+1]) * 100 for i in range(len(wins_p2))]
        
        episodes = range(len(win_rates_p1))
        axes[1].plot(episodes, win_rates_p1, label='Agent 1 Win Rate', color='red', alpha=0.8)
        axes[1].plot(episodes, win_rates_p2, label='Agent 2 Win Rate', color='blue', alpha=0.8)
        axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (balanced)')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Win Rate (%)')
        axes[1].set_title('Win Rate Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(f'training_progress_ep{episode}.png', dpi=100)
    plt.close()


if __name__ == "__main__":
    train_agents(num_episodes=5000, save_interval=500, visualize_interval=100)