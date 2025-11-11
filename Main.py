"""
Main entry point for 1D Combat Arena RL project
Self-play training with multi-player support
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='1D Combat Arena - Self-Play RL Training and Visualization'
    )
    
    parser.add_argument(
        'mode', 
        choices=['train', 'visualize', 'test'], 
        help='Mode: train agent, visualize gameplay, or test environment'
    )
    
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=5000,
        help='Number of training episodes (default: 5000)'
    )
    
    parser.add_argument(
        '--players', 
        type=int, 
        default=4,
        help='Number of players (default: 4, max: 8)'
    )
    
    parser.add_argument(
        '--games', 
        type=int, 
        default=3,
        help='Number of games to visualize (default: 3)'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='models/agent_self_play_final.pth',
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--fps', 
        type=int, 
        default=30,
        help='Frames per second for visualization (default: 30)'
    )
    
    parser.add_argument(
        '--renderer',
        choices=['pygame', 'matplotlib'],
        default='pygame',
        help='Visualization renderer (default: pygame)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("=" * 70)
        print(f"Starting Self-Play Training with {args.players} players")
        print("=" * 70)
        
        from Function.training.train import train_self_play
        train_self_play(
            num_episodes=args.episodes,
            num_players=args.players
        )
        
    elif args.mode == 'visualize':
        print("=" * 70)
        print(f"Visualizing {args.players}-Player Combat")
        print("=" * 70)
        
        if args.renderer == 'pygame':
            from Function.visualization.visualizer import visualize_trained_agent
            visualize_trained_agent(
                args.model,
                num_players=args.players,
                num_games=args.games,
                fps=args.fps
            )
        else:
            print("Matplotlib renderer not yet updated for multi-player.")
            print("Use --renderer pygame")
        
    elif args.mode == 'test':
        print("=" * 70)
        print("Testing Environment")
        print("=" * 70)
        
        from Function.core.game_env import Combat1DEnv
        env = Combat1DEnv(strip_length=300, num_players=args.players)
        
        print(f"Environment created successfully!")
        print(f"Number of players: {env.num_players}")
        print(f"Player IDs: {env.player_ids}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test reset
        observations, info = env.reset()
        print(f"\nInitial state:")
        for pid in env.player_ids:
            print(f"  {pid}: pos={info['positions'][pid]:.1f}, "
                  f"HP={info['hp'][pid]:.0f}, "
                  f"Mana={info['mana'][pid]:.0f}")
        
        # Test random actions
        print(f"\nTesting random actions for 10 steps...")
        for step in range(10):
            actions = {pid: env.action_space.sample() for pid in env.player_ids}
            observations, rewards, term, trunc, info = env.step(actions)
            
            alive_status = ', '.join([
                f"{pid}: {info['hp'][pid]:.0f}HP" 
                for pid in env.player_ids if env.alive[pid]
            ])
            print(f"Step {step+1}: {alive_status} | Alive: {info['num_alive']}")
            
            if term or trunc:
                break
        
        print("\nEnvironment test completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("1D Combat Arena - Self-Play Reinforcement Learning")
        print("=" * 70)
        print("\nUsage examples:")
        print("\n  TRAINING:")
        print("    python Main.py train                      # Train 2 players")
        print("    python Main.py train --players 4          # Train 4 players (battle royale)")
        print("    python Main.py train --episodes 10000     # Extended training")
        print("\n  VISUALIZATION:")
        print("    python Main.py visualize                  # Watch 2-player duels")
        print("    python Main.py visualize --players 4      # Watch 4-player battles")
        print("    python Main.py visualize --games 10       # Watch 10 games")
        print("    python Main.py visualize --fps 60         # Faster visualization")
        print("\n  TESTING:")
        print("    python Main.py test                       # Test environment")
        print("    python Main.py test --players 8           # Test 8 players")
        print("\nFor all options: python Main.py --help")
        print("=" * 70)
        sys.exit(0)
    
    main()