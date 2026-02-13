import pygame
import math
from constants import * 
from track import Track as track 
class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0.0  
        self.speed = 0.0
        self.width = 20
        self.height = 12
        self.time_alive = 0.0
        
        self.original_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.original_surface.fill((255, 0, 0))

    def update(self, throttle, steer):
        if throttle > 0:
            self.speed += throttle * ACCELERATION
        elif throttle < 0:
            self.speed += throttle * BRAKE 
        else:
            # Give small acceleration 
            self.speed += 0.02 

        # Drag
        self.speed -= DRAG * self.speed

        # Speed down to 0 methodology
        if abs(self.speed) < 0.1: self.speed = 0
        self.speed = max(0.0, min(self.speed, MAX_SPEED))

        # Steer strenght
        if self.speed > 0:
            steer_strength = min(1.0, self.speed / 1.0) 
            self.angle += steer * MAX_STEER * steer_strength

        # Position update
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y -= math.sin(rad) * self.speed

    # Draw the car
    def draw(self, screen):
        rotated = pygame.transform.rotate(self.original_surface, self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect)

        # Sensors

        if hasattr(self, 'sensor_data'):
            for i, angle_offset in enumerate([0, 45, -45, 90, -90]):
                rad = math.radians(self.angle + angle_offset)
                end_x = self.x + math.cos(rad) * self.sensor_data[i]
                end_y = self.y - math.sin(rad) * self.sensor_data[i]
                pygame.draw.line(screen, (255, 255, 0), (self.x, self.y), (end_x, end_y), 1)

    def get_corners(self):
        diag = math.sqrt((self.width/2)**2 + (self.height/2)**2)
        corners = []
        offsets = [30, 150, 210, 330] #points for the rectange 

        for angle_offset in offsets:
            rad = math.radians(self.angle + angle_offset)
            corner_x = self.x + math.cos(rad) * diag
            corner_y = self.y - math.sin(rad) * diag
            corners.append((corner_x, corner_y))
        return corners
    
    #rays - sensors
    def cast_rays(self, track):
        self.sensor_data = [] 
        sensor_angles = [0, 45, -45, 90, -90] 

        for angle_offset in sensor_angles:
            dist = 10
            while dist < 200: 
                dist += 2
                rad = math.radians(self.angle + angle_offset)
                px = self.x + math.cos(rad) * dist
                py = self.y - math.sin(rad) * dist
                
                color = track.get_color_at(px, py)
                
                # If not track == obstacle 
                if color != ASPHALT and color != CHECKPOINTS and color != FINISH: 
                    break
            
            self.sensor_data.append(dist)
        return self.sensor_data

    #Colision 
    def check_collision(self, track):
        for corner in self.get_corners():
            color = track.get_color_at(corner[0], corner[1])
            if color not in [ASPHALT, CHECKPOINTS, FINISH, OBSTACLES]: 
                return True
        return False