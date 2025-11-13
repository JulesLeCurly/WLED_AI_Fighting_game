import Function.wled.Main_wled as wled
import numpy as np
import time

wled_module = wled.Wled()
wled_module.connect()

start = time.time()

import Function.Animation.All_animation as Animation
import random

Frame = 0
dir = 1

player = 200

player_moving = 50

dir_rec = random.randint(-1, 1)
player_moving += dir_rec

LED_COUNT = 294

while (time.time() - start) < 20:
    time.sleep(1/30)
    Frame += 1
    led_strip = [(0, 0, 0)] * LED_COUNT

    
    led_strip[player] = (255, 0, 0)

    led_strip[100] = (0, 0, 255)

    
    if dir_rec == 0:
        led_strip[player_moving] = (255, 255, 0)
    else:
        led_strip = Animation.move_player_animation(led_strip, player_moving - dir_rec, (255, 255, 0), Frame, dir_rec)


    if Frame <= 10:
        led_strip = Animation.punch_animation(led_strip, player, 3, Frame, dir)
        
        led_strip = Animation.laser_animation(led_strip, 100, 10, Frame, dir)
    else:
        dir_rec = random.randint(-1, 1)
        player_moving += dir_rec
        Frame = 0
        dir *= -1
    
    led_strip[192] = (255, 0, 0)
    led_strip = Animation.shield_animation(led_strip, 192, -1)
    led_strip = Animation.shield_animation(led_strip, 192, 1)

    led_strip[169] = Animation.color_player_hit((0, 255, 0), Frame)
    


    led_strip = np.clip(np.array(led_strip), 0, 255)

    led_strip = np.array(led_strip) / 4
    wled_module.send_colors(led_strip.tolist())
print(time.time() - start)

wled_module.diconnect()