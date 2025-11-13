
def _get_cyclic_distance(pos1, pos2, strip_length):
    """Calculate shortest distance on cyclic world"""
    direct = abs(pos2 - pos1)
    wrap = strip_length - direct
    return min(direct, wrap)

def _get_direction(from_pos, to_pos, strip_length):
    """Get direction: -1 (left), +1 (right)"""
    direct = to_pos - from_pos
    wrap_right = (to_pos + strip_length) - from_pos
    wrap_left = to_pos - (from_pos + strip_length)
    
    min_dist = min(abs(direct), abs(wrap_right), abs(wrap_left))
    
    if abs(direct) == min_dist:
        return 1 if direct > 0 else -1
    elif abs(wrap_right) == min_dist:
        return 1
    else:
        return -1

def _get_closest_opponent(player_id, positions, alive, player_ids, strip_length):
    """Find closest alive opponent"""
    my_pos = positions[player_id]
    min_dist = float('inf')
    closest_pos = None
    
    for pid in player_ids:
        if pid != player_id and alive[pid]:
            dist = _get_cyclic_distance(my_pos, positions[pid])
            if dist < min_dist:
                min_dist = dist
                closest_pos = positions[pid]
    
    if closest_pos is None:
        return strip_length / 2, 0  # No opponents
    
    direction = _get_direction(my_pos, closest_pos)
    return min_dist, direction

def _get_n_closest_opponents(player_id, positions, alive, player_ids, strip_length, n=3):
    """Retourne les n adversaires vivants les plus proches d'un joueur donné."""
    my_pos = positions[player_id]
    distances = []

    # Calculer la distance à chaque adversaire vivant
    for pid in player_ids:
        if pid != player_id and alive[pid]:
            dist = _get_cyclic_distance(my_pos, positions[pid], strip_length)
            direction = _get_direction(my_pos, positions[pid], strip_length)
            distances.append((dist, direction, pid))

    # Trier les adversaires par distance croissante
    distances.sort(key=lambda x: x[0])

    # Garder seulement les n plus proches
    n_closest = distances[:n]

    # Si aucun adversaire vivant
    if not n_closest:
        return [(strip_length / 2, 0, None)]

    return n_closest