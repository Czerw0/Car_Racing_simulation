import torch
import torch.optim as optim
from env.car_env import CarEnv_basic
from models.car_net import CarNetwork
from torch.distributions import Categorical
import os
import pygame
import datetime
import random
import json
import logging_method

STATS_FILE = 'saves/stats.json'
BEST_MODEL_PATH = 'saves/best_model_basic.pth'
LATEST_MODEL = 'saves/latest_model_basic.pth'

# Load global best time 
def load_global_best():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                return json.load(f).get('global_best_lap', float('inf'))
        except: return float('inf')
    return float('inf')

#Safe global best time 
def save_global_best(t):
    with open(STATS_FILE, 'w') as f:
        json.dump({'global_best_lap': t}, f)


#Main training method 
def train():
    if not os.path.exists('saves'): os.makedirs('saves')
    if not os.path.exists('logs'): os.makedirs('logs')

    # Define enviroments
    env = CarEnv_basic()
    model = CarNetwork() 
    
    # Loading best model 
    global_best_lap = load_global_best()
    env.best_lap = global_best_lap
    if os.path.exists(BEST_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
            print(f"Loaded Model. Current Global Record: {global_best_lap:.2f}s")
        except: print("Starting fresh")

    #Optimizer - Adam 
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    print("Basic Training Start")

    for epoch in range(1, 10001):
        state = env.reset()
        log_probs, rewards = [], []
        done = False
        epoch_reward = 0
        start_tick = pygame.time.get_ticks()
        lap_time = (pygame.time.get_ticks() - start_tick) / 1000.0

        while not done:
            state_t = torch.from_numpy(state).float()
            logits = model(state_t)
            probs = torch.softmax(logits, dim=0)
            dist = Categorical(probs)
            
            # Epsilon - the random exploration 
            epsilon = max(0.05, 0.2 - (epoch / 2000))
            if random.random() < epsilon:
                action = torch.tensor(random.randint(0, 3))
            else:
                action = dist.sample()
            

            log_probs.append(dist.log_prob(action))
            state, reward, done = env.step(action.item())
            rewards.append(reward)
            epoch_reward += reward

            env.render(epoch=epoch, reward=epoch_reward)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    torch.save(model.state_dict(), LATEST_MODEL)
                    pygame.quit()
                    return

        # Learning - reinfomance
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            if r < 5000: r = max(min(r, 50), -50)
            R = r + 0.98 * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        policy_loss = []
        for lp, r in zip(log_probs, discounted_rewards):
            policy_loss.append(-lp * r)
        
        optimizer.zero_grad()
        if policy_loss:
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Record logic
        if env.best_lap < global_best_lap:
            global_best_lap = env.best_lap
            save_global_best(global_best_lap)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"NEW RECORD: {global_best_lap:.2f}s! Model Saved.")
        
        #Save for plotting
        logging_method.save_for_plotting(lap_time, global_best_lap, epoch)

        # Termianl loging 
        if epoch % 10 == 0:
            print(f"Ep: {epoch} | Rew: {epoch_reward:.1f} | Current: {lap_time}s | Margin: {(global_best_lap - lap_time):.2f}s")
            logging_method.log('training_basic_log.txt', f"Ep: {epoch} | Rew: {epoch_reward:.1f} | Current: {lap_time}s | Margin: {(global_best_lap - lap_time):.2f}s")
        

if __name__ == "__main__":
    train()