import Function.core.Main_game as main_game


game = main_game.Main_game(train=True, train_parms={"visualize_interval": 100, "save_interval": 500})

game.Lunch_game(NB_episodes=5000)