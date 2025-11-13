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