import pygame
import sys

import numpy as np

import Function.Animation.All_animation as Animation

import random
import time

# --- Constantes ---
LED_COUNT = 294
LED_SIZE = 10          # diamètre d'une LED simulée
HEIGHT = 60           # hauteur de la fenêtre


WIDTH = LED_COUNT * LED_SIZE

# --- Initialisation Pygame ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulation Ruban LED")
clock = pygame.time.Clock()

# --- Fonction pour afficher le ruban ---
def afficher_ruban(couleurs):
    """
    Affiche les LEDs selon la liste de tuples RGB fournie.
    Exemple: [(255,0,0), (0,255,0), (0,0,255), ...]
    """
    screen.fill((0, 0, 0))  # fond noir

    for i, couleur in enumerate(couleurs[:LED_COUNT]):
        x = i * LED_SIZE
        pygame.draw.rect(screen, couleur, (x, 0, LED_SIZE, HEIGHT))

    pygame.display.flip()

# --- Exemple d'utilisation ---
# Ici on crée une liste de couleurs en dégradé

# --- Boucle principale ---
Frame = 0
dir = 1

player = 200

player_moving = 50

dir_rec = random.randint(-1, 1)
player_moving += dir_rec
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
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

    afficher_ruban(led_strip.tolist())
    time.sleep(1/30)
