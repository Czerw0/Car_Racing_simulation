import torch
import numpy as np
import pygame
import os
from env.car_env import CarEnv_test  # Importujemy nowe środowisko testowe
from models.car_net import CarNetwork
import logging_method

def test():
    # 1. Konfiguracja
    MODEL_PATH = 'saves/best_model_time.pth'
    
    # 2. Inicjalizacja środowiska i modelu
    env = CarEnv_test()
    model = CarNetwork()
    
    # 3. Ładowanie wag modelu
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
            model.eval()
            print(f"--- Loaded Model: {MODEL_PATH} ---")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"!!! Error: File {MODEL_PATH} not found !!!")
        return

    # Pobieramy rekord z env (lub ustawiamy bazowy do porównania w logach)
    global_best_lap = 20.00  # Możesz tu wpisać swój rekord życiowy do porównania
    running = True
    epoch = 1

    while running:
        state = env.reset()
        done = False
        epoch_reward = 0
        
        while not done:
            state_t = torch.from_numpy(state).float()
            
            with torch.no_grad():
                logits = model(state_t)
                action = torch.argmax(logits).item()
            
            state, reward, done = env.step(action)
            epoch_reward += reward
            
            env.render(epoch=epoch) # Renderowanie z uproszczonego env

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    done = True

        # OBLICZENIA DO LOGA
        # Obliczamy czas trwania okrążenia na podstawie ticków z env
        lap_time = (pygame.time.get_ticks() - env.start_tick) / 1000.0
        margin = global_best_lap - lap_time

        # LOGOWANIE
        print(f"Test Run {epoch} Finished | Time: {lap_time:.2f}s | Reward: {epoch_reward:.1f}")
        
        # Wywołanie Twojej metody logowania (naprawiony cudzysłów i zmienne)
        log_msg = f"Ep: {epoch} | Rew: {epoch_reward:.1f} | Current: {lap_time:.2f}s | Margin: {margin:.2f}s"
        logging_method.log("test_log.txt", log_msg)

        epoch += 1
        pygame.time.wait(500)

    pygame.quit()

if __name__ == "__main__":
    test()