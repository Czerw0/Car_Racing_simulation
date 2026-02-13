import pygame
from constants import *

class Track:
    def __init__(self, track_name="Race_Track3"):
        self.current_track = track_name
        self.load_track(track_name)

    def load_track(self, track_name):
        self.current_track = track_name
        try:
            # Ładowanie grafiki z folderu env/
            self.image = pygame.image.load(f"env/{track_name}.png").convert()
            self.image = pygame.transform.scale(self.image, (1200, 800))
        except Exception as e:
            print(f"Błąd ładowania toru {track_name}: {e}")

    def draw(self, main_screen):
        main_screen.blit(self.image, (0, 0))
    
    def get_color_at(self, x, y):
        if 0 <= x < 1200 and 0 <= y < 800:
            color = self.image.get_at((int(x), int(y)))
            return (color.r, color.g, color.b)
        return (0, 0, 0)