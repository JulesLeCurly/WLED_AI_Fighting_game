import Function.wled.Main_wled as wled
import numpy as np
import time

wled_module = wled.Wled()
wled_module.connect()

start = time.time()

from typing import List, Tuple

from typing import List, Tuple
import time
import random
import math
from typing import List, Tuple

def shield_animation(
    led_strip: List[Tuple[int, int, int]],
    player_pos: int,
    direction: int = 1
) -> List[Tuple[int, int, int]]:
    """
    Affiche un bouclier magique animé à côté du joueur.
    Les couleurs varient dans le temps avec un effet sinusoïdal et aléatoire.
    """
    n = len(led_strip)
    new_strip = led_strip.copy()

    # Position du bouclier
    shield_pos = player_pos + direction
    if not (0 <= shield_pos < n):
        return new_strip  # hors du ruban

    # Temps actuel (pour animation fluide)
    t = time.time()

    # Variation sinusoïdale (0 → 1)
    pulse = (math.sin(t * 3.2) + 1) / 2  # rythme principal
    color_shift = (math.cos(t * 1.7) + 1) / 2  # teinte secondaire
    flicker = random.uniform(0.85, 1.5)  # aléa subtil

    # Couleurs dynamiques (mélange de bleu, turquoise et violet)
    r = int(80 * color_shift * flicker)
    g = int(150 + 50 * pulse * flicker)
    b = int(200 + 55 * (1 - color_shift) * flicker)

    # Mise à jour du pixel bouclier
    new_strip[shield_pos] = (r, g, b)

    return new_strip

def punch_animation(
    led_strip: List[Tuple[int, int, int]],
    player_pos: int,
    punch_length: int,
    frame: int,
    direction: int = 1  # +1 = droite, -1 = gauche
) -> List[Tuple[int, int, int]]:
    """
    Anime un coup de poing sur un ruban LED 1D.
    frame ∈ [1, 10]
    direction: +1 pour droite, -1 pour gauche
    """
    n = len(led_strip)

    # Progression du coup
    progress = (frame - 1) / 9  # de 0 à 1
    reach = int(progress * punch_length)

    # Déterminer la zone active du coup selon la direction
    if direction == 1:  # droite
        start = player_pos + 1
        end = player_pos + reach + 1
    else:  # gauche
        start = player_pos - 1
        end = player_pos - reach - 1

    # Couleur selon la frame (inchangée)
    if frame <= 4:
        color = (
            int(255 * (1 - progress * 0.5)),  # R
            int(200 - 100 * progress),        # G
            0
        )
    elif frame <= 8:
        color = (
            255,
            int(50 - 50 * (progress - 0.4)),
            0
        )
    else:
        color = (255, 255, 255)  # Impact final (blanc vif)

    # Appliquer la couleur sur la zone du coup
    if direction == 1:
        indices = range(start, end)
    else:
        indices = range(start, end, -1)

    for i in indices:
        if 0 <= i < n:  # éviter les débordements
            fade = 1.0 - abs((i - end) / (punch_length + 1))
            fade = max(0.0, min(fade, 1.0))
            led_strip[i] = (
                int(color[0] * fade),
                int(color[1] * fade),
                int(color[2] * fade)
            )

    return led_strip

import time
import random
import math
from typing import List, Tuple

def laser_animation(
    led_strip: List[Tuple[int, int, int]],
    player_pos: int,
    laser_length: int,
    frame: int,
    direction: int = 1
) -> List[Tuple[int, int, int]]:
    """
    Anime un laser magique violet sur 10 frames.
    Le laser s'étend progressivement et scintille, puis disparaît à la dernière frame.
    """
    n = len(led_strip)

    # Disparition immédiate après la frame finale
    if frame >= 10:
        return led_strip

    # Progression du laser
    progress = (frame - 1) / 9  # 0 → 1
    reach = int(progress * laser_length)

    # Déterminer les indices du laser selon la direction
    if direction == 1:
        start = player_pos + 1
        end = min(player_pos + reach + 1, n)
        indices = range(start, end)
    else:
        start = player_pos - 1
        end = max(player_pos - reach - 1, -1)
        indices = range(start, end, -1)

    # Temps actuel pour variation continue
    t = time.time()

    for i in indices:
        if 0 <= i < n:
            # Variation aléatoire et sinusoïdale
            flicker = random.uniform(0, 2)
            wave = (math.sin(t * 6 + i * 0.5) + 1) / 2


            # Base violette dynamique
            r = int(180 + 40 * wave * flicker)   # rouge profond, pulsant
            g = int(0 + 40 * (1 - wave) * flicker)  # soupçon de turquoise
            b = int(220 + 35 * flicker)          # violet/bleu électrique

            
            if random.randint(0, 10) < 2:
                
                # Base violette dynamique
                r = int(r * 2)   # rouge profond, pulsant
                g = int(g / 2)  # soupçon de turquoise
                b = int(b / 2)          # violet/bleu électrique


            # Intensité centrale (plus lumineux vers la fin du laser)
            fade = 1.0 - abs((i - (start if direction == 1 else start)) / (laser_length + 1))
            fade = max(0.5, fade)

            led_strip[i] = (
                int(r * fade),
                int(g * fade),
                int(b * fade)
            )

    return led_strip

from typing import List, Tuple

def move_player_animation(
    led_strip: List[Tuple[int, int, int]],
    player_pos: int,
    color_player: Tuple[int, int, int],
    frame: int,
    direction: int = 1
) -> List[Tuple[int, int, int]]:
    """
    Anime le déplacement fluide du joueur sur 10 frames.
    Le joueur se déplace d'un pixel avec un effet de fondu (fade in/out).
    """
    n = len(led_strip)

    # Positions de départ et d’arrivée
    start = player_pos
    end = player_pos + direction

    # Sécurité (éviter de sortir du ruban)
    if not (0 <= start < n):
        return led_strip
    if not (0 <= end < n):
        end = start  # si on dépasse, le joueur reste sur place

    # Progression 0 → 1
    progress = (frame - 1) / 9

    # Intensité : départ s'éteint, arrivée s'allume
    fade_out = 1.0 - progress
    fade_in = progress

    # Application des couleurs avec intensité
    start_color = tuple(int(c * fade_out) for c in color_player)
    end_color = tuple(int(c * fade_in) for c in color_player)

    # Mise à jour du ruban
    led_strip[start] = start_color
    led_strip[end] = end_color

    return led_strip

def color_player_hit(color_player: Tuple[int, int, int], frame: int) -> Tuple[int, int, int]:
    """
    Anime la couleur du joueur lorsqu'il est touché par un coup de poing.
    """
    if frame in [1, 2, 5, 6, 9, 10]:
        return (0, 0, 0)
    else:
        return color_player