import socket
import numpy as np
from typing import List, Tuple


class Wled():
    def __init__(self) -> None:
        self.WLED_MODULES = [
            {"name": "Armoire", "ip": "192.168.1.28", "leds": 141},
            {"name": "Desktop", "ip": "192.168.1.27", "leds": 153},
        ]
        self.UDP_PORT = 65505
        self.PROTO_DRGB = 2

        self.total_leds = sum(m["leds"] for m in self.WLED_MODULES)
    
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def diconnect(self):
        self.sock.close()
    def send(self, m, data):
        self.sock.sendto(data, (m["ip"], self.UDP_PORT))
    
    def send_colors(self, strip_colors: List[Tuple[int, int, int]], reorder=(0,1,2)):
        
        if len(strip_colors) != self.total_leds:
            raise ValueError(f"Il faut {self.total_leds} couleurs (tu en as {len(strip_colors)})")

        arr = np.array(strip_colors, dtype=np.uint8)
        arr = arr[:, list(reorder)]

        offset = 0
        for m in self.WLED_MODULES:
            count = m["leds"]
            segment = arr[offset:offset + count]
            if m["name"] == "Armoire":
                segment = segment[::-1]
            offset += count

            # Préparer le packet UDP : protocole + timeout + données couleurs
            timeout = 2  # seconde
            header = bytes([self.PROTO_DRGB, timeout])
            # segment flatten en bytes
            body = segment.tobytes()
            packet = header + body

            self.send(m, packet)