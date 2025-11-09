"""
Main entry point for 1D Combat Arena RL project
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='1D Combat Arena - RL Training and Visualization')
    parser.add_argument('mode', choices=['train', 'visualize', 'test'], 
                        help='Mode: train agents, visualize gameplay, or test environment')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes (default: 5000)')
    parser.add_argument('--games', type=int, default=3,
                        help='Number of games to visualize (default: 3)')
    parser.add_argument('--agent1', type=str, default='models/agent1_final.pth',
                        help='Path to agent 1 model')
    parser.add_argument('--agent2', type=str, default='models/agent2_final.pth',
                        help='Path to agent 2 model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training mode...")
        from Function.training.train import train_agents
        train_agents(num_episodes=args.episodes)
        
    elif args.mode == 'visualize':
        print("Starting visualization mode...")
        from Function.visualization.visualizer import visualize_trained_agents
        visualize_trained_agents(args.agent1, args.agent2, num_games=args.games)
        
    elif args.mode == 'test':
        print("Testing environment...")
        from Function.core.game_env import Combat1DEnv
        env = Combat1DEnv(strip_length=300)
        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Run a quick test
        (obs_p1, obs_p2), info = env.reset()
        print(f"\nInitial state:")
        print(f"Player 1 observation: {obs_p1}")
        print(f"Player 2 observation: {obs_p2}")
        print(f"Info: {info}")
        
        # Test random actions
        print("\nTesting random actions for 10 steps...")
        for step in range(10):
            action_p1 = env.action_space.sample()
            action_p2 = env.action_space.sample()
            (obs_p1, obs_p2), (r_p1, r_p2), term, trunc, info = env.step((action_p1, action_p2))
            print(f"Step {step+1}: P1 HP={info['p1_hp']:.0f}, P2 HP={info['p2_hp']:.0f}")
            if term or trunc:
                break
        
        print("\nEnvironment test completed successfully!")


if __name__ == "__main__":
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("=" * 60)
        print("1D Combat Arena - Reinforcement Learning Battle")
        print("=" * 60)
        print("\nUsage examples:")
        print("  python Main.py train                    # Train agents")
        print("  python Main.py train --episodes 10000   # Train for 10k episodes")
        print("  python Main.py visualize                # Visualize trained agents")
        print("  python Main.py visualize --games 5      # Watch 5 games")
        print("  python Main.py test                     # Test environment")
        print("\nFor more options: python Main.py --help")
        print("=" * 60)
        sys.exit(0)
    
    main()