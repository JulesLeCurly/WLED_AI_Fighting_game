from Function.other.All_basic_function import read_yml

from Function.core.Game_env import Combat1v1Env
from collections import deque
from Function.torch.Train_function import plot_training_progress

import os

import numpy as np

class Main_game():

    def __init__(self, train=False, train_parms=None):
        self.train = train
        self.train_parms = train_parms

        self.Config = read_yml("Config/Game_env.yml")


        self.env = Combat1v1Env(strip_length=self.Config["World_length"], num_players=self.Config["num_players"], train=self.train)


    def Lunch_game(self, NB_episodes):
        episode_rewards = []
        episode_lengths = []
        survival_times = {pid: deque(maxlen=100) for pid in self.env.player_ids}
        for episode in range(1, NB_episodes + 1):
            log_player_rewards, log_survival_times, total_steps, NB_Survivors = self.env.Episode(self.env)
            avg_reward = np.mean(list(log_player_rewards.values()))
            episode_rewards.append(avg_reward)
            episode_lengths.append(total_steps)
        
            # Record survival times
            for player_id in self.env.player_ids:
                survival_times[player_id].append(log_survival_times[player_id])

            if self.train:
                # Print progress
                if episode % 10 == 0:
                    avg_reward_100 = np.mean(episode_rewards[-100:])
                    avg_length = np.mean(episode_lengths[-100:])
                    avg_survival = {
                        pid: np.mean(log_survival_times[pid]) 
                        for pid in self.env.player_ids
                    }
                    
                    print(f"Ep {episode:4d} | "
                        f"Steps: {total_steps:3d} | "
                        f"Avg Reward: {avg_reward_100:7.2f} | "
                        f"Avg Length: {avg_length:5.1f} | "
                        f"Survivors: {NB_Survivors} | "
                        f"ε: {self.env.agent.epsilon:.3f}")
                    
                    if episode % 50 == 0:
                        print(f"  Avg Survival: {', '.join([f'{pid}: {avg_survival[pid]:.0f}' for pid in self.env.player_ids])}")
                
                # Visualize progress
                if episode % self.train_parms["visualize_interval"] == 0:
                    plot_training_progress(episode_rewards, episode_lengths, survival_times, episode)
                
                # Save model
                if episode % self.train_parms["save_interval"] == 0:
                    os.makedirs("models", exist_ok=True)
                    self.env.agent.save(f"models/agent_self_play_ep{episode}.pth")
                    print(f"Model saved at episode {episode}")
            
        # Final save
        self.env.agent.save("models/agent_self_play_final.pth")
        print("\nTraining completed!")



from Function.other.All_basic_function import read_yml
from Function.core.Game_env import Combat1v1Env
from collections import deque
import numpy as np
import os
import time


class ImprovedTraining:
    """Improved training with better logging and hyperparameters"""
    
    def __init__(self, train=True):
        self.train = train
        self.Config = read_yml("Config/Game_env.yml")
        
        # Create environment
        self.env = Combat1v1Env(
            strip_length=self.Config["World_length"],
            num_players=self.Config["num_players"],
            train=self.train
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.survival_times = {pid: deque(maxlen=100) for pid in self.env.player_ids}
        self.win_rates = {pid: deque(maxlen=100) for pid in self.env.player_ids}
        
        # Best model tracking
        self.best_avg_reward = float('-inf')
        self.best_model_path = "models/best_agent.pth"
    
    def train_agent(self, num_episodes=5000, visualize_interval=100, save_interval=500):
        """Train agent with improved logging"""
        print("=" * 60)
        print(f"Starting Training: {num_episodes} episodes")
        print(f"State size: {self.env.observation_space.shape[0]}")
        print(f"Action size: {self.env.action_space.n}")
        print(f"Device: {self.env.agent.device}")
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            # Run episode
            log_player_rewards, log_survival_times, total_steps, nb_survivors = self.env.Episode()
            
            # Calculate metrics
            avg_reward = np.mean(list(log_player_rewards.values()))
            self.episode_rewards.append(avg_reward)
            self.episode_lengths.append(total_steps)
            
            # Record survival times and wins
            for player_id in self.env.player_ids:
                self.survival_times[player_id].append(log_survival_times[player_id])
                self.win_rates[player_id].append(1 if self.env.alive[player_id] else 0)
            
            # Logging every 10 episodes
            if episode % 10 == 0:
                avg_reward_100 = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else avg_reward
                avg_length = np.mean(self.episode_lengths[-100:])
                
                # Calculate win rates
                win_rate_str = " | ".join([
                    f"{pid}: {np.mean(self.win_rates[pid])*100:.1f}%"
                    for pid in self.env.player_ids
                ])
                
                print(f"\nEp {episode:4d} | "
                      f"Reward: {avg_reward_100:7.2f} | "
                      f"Steps: {avg_length:5.1f} | "
                      f"Survivors: {nb_survivors} | "
                      f"ε: {self.env.agent.epsilon:.3f}")
                print(f"  Win rates: {win_rate_str}")
                
                # Loss tracking
                if self.env.agent.losses:
                    avg_loss = np.mean(self.env.agent.losses[-100:])
                    print(f"  Avg Loss: {avg_loss:.4f}")
            
            # Detailed logging every 50 episodes
            if episode % 50 == 0:
                avg_survival = {
                    pid: np.mean(self.survival_times[pid])
                    for pid in self.env.player_ids
                }
                survival_str = " | ".join([
                    f"{pid}: {avg_survival[pid]:.1f}"
                    for pid in self.env.player_ids
                ])
                print(f"  Avg Survival: {survival_str}")
                
                elapsed_time = time.time() - start_time
                eps_per_sec = episode / elapsed_time
                eta_seconds = (num_episodes - episode) / eps_per_sec
                eta_minutes = eta_seconds / 60
                print(f"  Speed: {eps_per_sec:.2f} eps/s | ETA: {eta_minutes:.1f} min")
            
            # Visualization
            if episode % visualize_interval == 0:
                self._plot_training_progress(episode)
            
            # Save model periodically
            if episode % save_interval == 0:
                os.makedirs("models", exist_ok=True)
                self.env.agent.save(f"models/agent_ep{episode}.pth")
                print(f"✓ Model saved at episode {episode}")
            
            # Save best model
            if len(self.episode_rewards) >= 100:
                current_avg = np.mean(self.episode_rewards[-100:])
                if current_avg > self.best_avg_reward:
                    self.best_avg_reward = current_avg
                    os.makedirs("models", exist_ok=True)
                    self.env.agent.save(self.best_model_path)
                    print(f"✓ New best model! Avg reward: {self.best_avg_reward:.2f}")
        
        # Final save
        self.env.agent.save("models/agent_final.pth")
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
        print(f"Best average reward: {self.best_avg_reward:.2f}")
        print("=" * 60)
    
    def _plot_training_progress(self, episode):
        """Plot training statistics"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            window = 100
            os.makedirs('training_progress', exist_ok=True)
            
            # Plot 1: Average Reward
            if len(self.episode_rewards) >= window:
                avg_rewards = [np.mean(self.episode_rewards[max(0, i-window):i+1]) 
                              for i in range(len(self.episode_rewards))]
                axes[0].plot(avg_rewards, color='purple', linewidth=2)
                axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                axes[0].set_xlabel('Episode')
                axes[0].set_ylabel('Average Reward')
                axes[0].set_title(f'Training Progress - Episode {episode}')
                axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Episode Length
            if len(self.episode_lengths) >= window:
                avg_lengths = [np.mean(self.episode_lengths[max(0, i-window):i+1]) 
                              for i in range(len(self.episode_lengths))]
                axes[1].plot(avg_lengths, color='orange', linewidth=2)
                axes[1].set_xlabel('Episode')
                axes[1].set_ylabel('Episode Length (steps)')
                axes[1].set_title('Combat Duration')
                axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Win Rates
            colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan']
            for i, (player_id, win_history) in enumerate(self.win_rates.items()):
                if len(win_history) >= window:
                    win_rate_avg = [np.mean(list(win_history)[max(0, j-window):j+1]) * 100
                                   for j in range(len(win_history))]
                    axes[2].plot(win_rate_avg, label=player_id, 
                               color=colors[i % len(colors)], 
                               linewidth=2, alpha=0.8)
            
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Win Rate (%)')
            axes[2].set_title('Player Performance Balance')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim([0, 100])
            
            plt.tight_layout()
            plt.savefig(f'training_progress/training_ep{episode}.png', dpi=100)
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not plot progress: {e}")


def main():
    """Main training script"""
    trainer = ImprovedTraining(train=True)
    
    # Training hyperparameters
    NUM_EPISODES = 5000
    VISUALIZE_INTERVAL = 100
    SAVE_INTERVAL = 500
    
    trainer.train_agent(
        num_episodes=NUM_EPISODES,
        visualize_interval=VISUALIZE_INTERVAL,
        save_interval=SAVE_INTERVAL
    )


if __name__ == "__main__":
    main()