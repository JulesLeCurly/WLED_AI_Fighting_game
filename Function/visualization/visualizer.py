import pygame
import numpy as np
from Function.core.game_env import Combat1DEnv
from Function.agents.dqn_agent import DQNAgent

class PygameVisualizer:
    """
    Real-time game visualization using Pygame
    Much smoother than matplotlib for animations
    """
    
    def __init__(self, env, agent, fps=30):
        self.env = env
        self.agent = agent
        self.fps = fps
        
        # Initialize Pygame
        pygame.init()
        
        # Screen dimensions
        self.width = 1400
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('1D Combat Arena - RL Battle')
        
        # Clock for FPS control
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Colors
        self.bg_color = (20, 20, 30)
        self.text_color = (255, 255, 255)
        self.grid_color = (50, 50, 60)
        
        # LED strip visualization area
        self.strip_x = 50
        self.strip_y = 100
        self.strip_width = 1300
        self.strip_height = 80
        self.pixel_width = self.strip_width / env.strip_length
        
        # Stats area
        self.stats_y = 220
        self.stats_height = 200
        
        # HP History
        self.history_y = 450
        self.history_height = 300
        self.hp_history = {pid: [] for pid in env.player_ids}
        
        # Action names
        self.action_names = ['LEFT', 'RIGHT', 'SHORT ATK', 'LONG ATK', 'REST']
    
    def draw_led_strip(self, strip_data):
        """Draw the LED strip"""
        for i in range(self.env.strip_length):
            x = self.strip_x + i * self.pixel_width
            color = tuple(strip_data[i].astype(int))
            pygame.draw.rect(
                self.screen,
                color,
                (x, self.strip_y, self.pixel_width + 1, self.strip_height)
            )
        
        # Border
        pygame.draw.rect(
            self.screen,
            self.text_color,
            (self.strip_x - 2, self.strip_y - 2, 
             self.strip_width + 4, self.strip_height + 4),
            2
        )
        
        # Title
        title = self.font_medium.render('LED STRIP (300 pixels)', True, self.text_color)
        self.screen.blit(title, (self.strip_x, self.strip_y - 40))
    
    def draw_player_stats(self, info):
        """Draw player stats bars"""
        y_offset = self.stats_y
        spacing = 80
        
        for i, player_id in enumerate(self.env.player_ids):
            if not self.env.alive[player_id]:
                continue
            
            y = y_offset + i * spacing
            
            # Player label
            color = self.env.player_colors[player_id]
            label = self.font_small.render(f'{player_id.upper()}', True, color)
            self.screen.blit(label, (self.strip_x - 30, y + 5))
            
            # HP bar
            hp_width = int((info['hp'][player_id] / self.env.max_hp) * 400)
            pygame.draw.rect(self.screen, color, (self.strip_x + 50, y, hp_width, 25))
            pygame.draw.rect(self.screen, self.grid_color, (self.strip_x + 50, y, 400, 25), 2)
            hp_text = self.font_small.render(f"HP: {info['hp'][player_id]:.0f}", True, self.text_color)
            self.screen.blit(hp_text, (self.strip_x + 460, y + 2))
            
            # Mana bar
            mana_width = int((info['mana'][player_id] / self.env.max_mana) * 400)
            pygame.draw.rect(self.screen, (100, 200, 255), (self.strip_x + 50, y + 30, mana_width, 25))
            pygame.draw.rect(self.screen, self.grid_color, (self.strip_x + 50, y + 30, 400, 25), 2)
            mana_text = self.font_small.render(f"Mana: {info['mana'][player_id]:.0f}", True, self.text_color)
            self.screen.blit(mana_text, (self.strip_x + 460, y + 32))
            
            # Last action
            if info['last_actions'][player_id] is not None:
                action = self.action_names[info['last_actions'][player_id]]
                action_text = self.font_small.render(f"Action: {action}", True, (255, 255, 0))
                self.screen.blit(action_text, (self.strip_x + 600, y + 15))
    
    def draw_hp_history(self):
        """Draw HP history graph"""
        x = self.strip_x
        y = self.history_y
        width = self.strip_width
        height = self.history_height
        
        # Background
        pygame.draw.rect(self.screen, (30, 30, 40), (x, y, width, height))
        pygame.draw.rect(self.screen, self.grid_color, (x, y, width, height), 2)
        
        # Title
        title = self.font_medium.render('HP HISTORY', True, self.text_color)
        self.screen.blit(title, (x, y - 40))
        
        # Grid lines
        for i in range(5):
            grid_y = y + (height // 4) * i
            pygame.draw.line(self.screen, self.grid_color, (x, grid_y), (x + width, grid_y), 1)
        
        # Draw HP lines
        colors_rgb = {
            'p1': (255, 0, 0),
            'p2': (0, 0, 255),
            'p3': (0, 255, 0),
            'p4': (255, 255, 0),
            'p5': (255, 0, 255),
            'p6': (0, 255, 255),
        }
        
        for player_id, history in self.hp_history.items():
            if len(history) < 2:
                continue
            
            color = colors_rgb.get(player_id, (255, 255, 255))
            points = []
            
            for i, hp in enumerate(history[-200:]):  # Last 200 steps
                px = x + (i / max(len(history[-200:]), 1)) * width
                py = y + height - (hp / 100) * height
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 3)
            
            # Legend
            legend_text = self.font_small.render(player_id.upper(), True, color)
            legend_x = x + width - 100 + list(self.hp_history.keys()).index(player_id) * 50
            self.screen.blit(legend_text, (legend_x, y - 35))
    
    def draw_game_info(self, info):
        """Draw general game information"""
        # Step counter
        step_text = self.font_medium.render(f"Step: {info['step']}/{self.env.max_steps}", 
                                            True, self.text_color)
        self.screen.blit(step_text, (self.width - 250, 30))
        
        # Alive counter
        alive_text = self.font_medium.render(f"Alive: {info['num_alive']}/{self.env.num_players}", 
                                             True, (0, 255, 0))
        self.screen.blit(alive_text, (self.width - 250, 70))
    
    def run_game(self, num_games=3):
        """Run and visualize games"""
        running = True
        
        for game_num in range(1, num_games + 1):
            if not running:
                break
            
            print(f"\n{'='*60}")
            print(f"Game {game_num}/{num_games}")
            print('='*60)
            
            # Reset
            observations, info = self.env.reset()
            self.hp_history = {pid: [info['hp'][pid]] for pid in self.env.player_ids}
            
            done = False
            
            while not done and running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            # Pause
                            paused = True
                            while paused:
                                for e in pygame.event.get():
                                    if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                                        paused = False
                                    elif e.type == pygame.QUIT:
                                        running = False
                                        paused = False
                
                # Select actions (same agent for all players)
                actions = {}
                for player_id in self.env.player_ids:
                    if self.env.alive[player_id]:
                        obs = observations[player_id]
                        actions[player_id] = self.agent.select_action(obs, training=False)
                
                # Step
                observations, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                # Update history
                for player_id in self.env.player_ids:
                    self.hp_history[player_id].append(info['hp'][player_id])
                
                # Draw everything
                self.screen.fill(self.bg_color)
                
                strip_data = self.env.render()
                self.draw_led_strip(strip_data)
                self.draw_player_stats(info)
                self.draw_hp_history()
                self.draw_game_info(info)
                
                pygame.display.flip()
                self.clock.tick(self.fps)
            
            # Game over screen
            if running:
                self.screen.fill(self.bg_color)
                
                # Find winner(s)
                max_hp = max(info['hp'].values())
                winners = [pid for pid in self.env.player_ids if info['hp'][pid] == max_hp]
                
                if len(winners) == 1:
                    winner_text = self.font_large.render(f"WINNER: {winners[0].upper()}", 
                                                         True, self.env.player_colors[winners[0]])
                else:
                    winner_text = self.font_large.render("DRAW!", True, (255, 255, 0))
                
                text_rect = winner_text.get_rect(center=(self.width // 2, self.height // 2))
                self.screen.blit(winner_text, text_rect)
                
                # Stats
                stats_text = self.font_medium.render(
                    f"Duration: {info['step']} steps", 
                    True, self.text_color
                )
                stats_rect = stats_text.get_rect(center=(self.width // 2, self.height // 2 + 60))
                self.screen.blit(stats_text, stats_rect)
                
                pygame.display.flip()
                pygame.time.wait(3000)  # Wait 3 seconds
        
        pygame.quit()


def visualize_trained_agent(model_path, num_players=2, num_games=3, fps=30):
    """Load trained agent and visualize gameplay"""
    env = Combat1DEnv(strip_length=300, num_players=num_players)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0  # No exploration
    
    visualizer = PygameVisualizer(env, agent, fps=fps)
    visualizer.run_game(num_games=num_games)


if __name__ == "__main__":
    # Visualize 2-player duel
    visualize_trained_agent(
        "models/agent_self_play_final.pth",
        num_players=2,
        num_games=5,
        fps=30
    )
    
    # Or visualize 4-player battle royale
    # visualize_trained_agent(
    #     "models/agent_self_play_final.pth",
    #     num_players=4,
    #     num_games=3,
    #     fps=30
    # )