import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
from Function.game_env import Combat1DEnv
from Function.dqn_agent import DQNAgent

class GameVisualizer:
    """Real-time visualization of the game"""
    
    def __init__(self, env, agent1, agent2):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        
        # Setup figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 8))
        self.fig.suptitle('1D Combat Arena - RL Agents Battle', fontsize=16, fontweight='bold')
        
        # LED strip axis
        self.ax_strip = self.axes[0]
        self.ax_strip.set_xlim(0, env.strip_length)
        self.ax_strip.set_ylim(0, 1)
        self.ax_strip.set_title('LED Strip (300 pixels)')
        self.ax_strip.set_yticks([])
        
        # Stats axis
        self.ax_stats = self.axes[1]
        self.ax_stats.set_xlim(0, 100)
        self.ax_stats.set_ylim(0, 5)
        self.ax_stats.set_title('Player Stats')
        self.ax_stats.set_yticks([])
        
        # History axis
        self.ax_history = self.axes[2]
        self.ax_history.set_title('HP History')
        self.ax_history.set_xlabel('Step')
        self.ax_history.set_ylabel('HP')
        self.ax_history.set_ylim(0, 100)
        
        # History data
        self.hp_history_p1 = []
        self.hp_history_p2 = []
        
        plt.tight_layout()
    
    def update_display(self, info):
        """Update all visualizations"""
        # Clear axes
        self.ax_strip.clear()
        self.ax_stats.clear()
        
        # LED strip visualization
        strip = self.env.render()
        
        for i in range(len(strip)):
            color = strip[i] / 255.0
            self.ax_strip.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='none'))
        
        self.ax_strip.set_xlim(0, self.env.strip_length)
        self.ax_strip.set_ylim(0, 1)
        self.ax_strip.set_title('LED Strip (300 pixels)', fontsize=12, fontweight='bold')
        self.ax_strip.set_yticks([])
        self.ax_strip.set_xticks([0, 75, 150, 225, 300])
        
        # Stats visualization
        self.ax_stats.set_xlim(0, 100)
        self.ax_stats.set_ylim(0, 6)
        
        # Player 1 stats (Red)
        p1_hp_bar = Rectangle((0, 4.5), info['p1_hp'], 0.4, facecolor='red', edgecolor='black', linewidth=2)
        p1_mana_bar = Rectangle((0, 3.8), info['p1_mana'], 0.4, facecolor='orange', edgecolor='black', linewidth=2)
        self.ax_stats.add_patch(p1_hp_bar)
        self.ax_stats.add_patch(p1_mana_bar)
        self.ax_stats.text(-5, 4.7, 'P1 HP:', fontsize=10, ha='right', va='center', fontweight='bold')
        self.ax_stats.text(-5, 4.0, 'P1 Mana:', fontsize=10, ha='right', va='center', fontweight='bold')
        self.ax_stats.text(102, 4.7, f"{info['p1_hp']:.0f}", fontsize=10, ha='left', va='center')
        self.ax_stats.text(102, 4.0, f"{info['p1_mana']:.0f}", fontsize=10, ha='left', va='center')
        
        # Player 2 stats (Blue)
        p2_hp_bar = Rectangle((0, 2.5), info['p2_hp'], 0.4, facecolor='blue', edgecolor='black', linewidth=2)
        p2_mana_bar = Rectangle((0, 1.8), info['p2_mana'], 0.4, facecolor='cyan', edgecolor='black', linewidth=2)
        self.ax_stats.add_patch(p2_hp_bar)
        self.ax_stats.add_patch(p2_mana_bar)
        self.ax_stats.text(-5, 2.7, 'P2 HP:', fontsize=10, ha='right', va='center', fontweight='bold')
        self.ax_stats.text(-5, 2.0, 'P2 Mana:', fontsize=10, ha='right', va='center', fontweight='bold')
        self.ax_stats.text(102, 2.7, f"{info['p2_hp']:.0f}", fontsize=10, ha='left', va='center')
        self.ax_stats.text(102, 2.0, f"{info['p2_mana']:.0f}", fontsize=10, ha='left', va='center')
        
        # Action labels
        actions = ['LEFT', 'RIGHT', 'SHORT ATK', 'LONG ATK']
        p1_action = actions[info['p1_action']] if info['p1_action'] is not None else 'NONE'
        p2_action = actions[info['p2_action']] if info['p2_action'] is not None else 'NONE'
        
        self.ax_stats.text(50, 5.5, f'Last Action: {p1_action}', fontsize=9, ha='center', color='red', fontweight='bold')
        self.ax_stats.text(50, 1.2, f'Last Action: {p2_action}', fontsize=9, ha='center', color='blue', fontweight='bold')
        
        self.ax_stats.set_title(f'Player Stats - Step {info["step"]}', fontsize=12, fontweight='bold')
        self.ax_stats.set_yticks([])
        self.ax_stats.set_xticks([])
        
        # HP history
        self.hp_history_p1.append(info['p1_hp'])
        self.hp_history_p2.append(info['p2_hp'])
        
        self.ax_history.clear()
        if len(self.hp_history_p1) > 1:
            self.ax_history.plot(self.hp_history_p1, color='red', linewidth=2, label='Player 1')
            self.ax_history.plot(self.hp_history_p2, color='blue', linewidth=2, label='Player 2')
        self.ax_history.set_title('HP History', fontsize=12, fontweight='bold')
        self.ax_history.set_xlabel('Step')
        self.ax_history.set_ylabel('HP')
        self.ax_history.set_ylim(0, 100)
        self.ax_history.legend()
        self.ax_history.grid(True, alpha=0.3)
        
        plt.pause(0.03)  # 30 FPS
    
    def run_game(self, num_games=1):
        """Run and visualize games"""
        plt.ion()
        
        for game_num in range(1, num_games + 1):
            print(f"\n{'='*60}")
            print(f"Starting Game {game_num}/{num_games}")
            print('='*60)
            
            # Reset
            (obs_p1, obs_p2), info = self.env.reset()
            self.hp_history_p1 = [info['p1_hp']]
            self.hp_history_p2 = [info['p2_hp']]
            
            done = False
            step = 0
            
            while not done:
                # Select actions (no exploration, use trained policy)
                action_p1 = self.agent1.select_action(obs_p1, training=False)
                action_p2 = self.agent2.select_action(obs_p2, training=False)
                
                # Step
                (obs_p1, obs_p2), rewards, terminated, truncated, info = self.env.step((action_p1, action_p2))
                done = terminated or truncated
                
                # Visualize
                self.update_display(info)
                
                step += 1
            
            # Game over
            print(f"\nGame {game_num} finished after {step} steps")
            if info['p1_hp'] > info['p2_hp']:
                print("WINNER: Player 1 (Red)")
            elif info['p2_hp'] > info['p1_hp']:
                print("WINNER: Player 2 (Blue)")
            else:
                print("RESULT: Draw")
            print(f"Final HP - P1: {info['p1_hp']:.0f}, P2: {info['p2_hp']:.0f}")
            
            if game_num < num_games:
                plt.pause(2)  # Pause between games
        
        plt.ioff()
        plt.show()


def visualize_trained_agents(agent1_path, agent2_path, num_games=3):
    """Load trained agents and visualize their gameplay"""
    env = Combat1DEnv(strip_length=300)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)
    
    agent1.load(agent1_path)
    agent2.load(agent2_path)
    
    agent1.epsilon = 0  # No exploration
    agent2.epsilon = 0
    
    visualizer = GameVisualizer(env, agent1, agent2)
    visualizer.run_game(num_games=num_games)


if __name__ == "__main__":
    # Example: visualize trained agents
    visualize_trained_agents(
        "models/agent1_final.pth",
        "models/agent2_final.pth",
        num_games=5
    )