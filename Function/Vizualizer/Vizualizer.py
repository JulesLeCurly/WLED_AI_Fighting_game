
import Function.wled.Main_wled as wled


class Vizualizer:
    def __init__(self, vizualizer_mode):
        self.vizualizer_mode = vizualizer_mode
        if self.vizualizer_mode == "wled":
            self.wled_module = wled.Wled()
            self.wled_module.connect()
    

    def show(self, All_game_actions, positions, shield_on_map):
        pass