import torch
import torch.optim as optim
from env.car_env import CarEnv_time
from models.car_net import CarNetwork
from torch.distributions import Categorical
import os, pygame, random, json
import logging_method

# Pliki
STATS_FILE = 'saves/stats.json'
BEST_MODEL_PATH = 'saves/best_model_time.pth'

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_stats(stats):
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=4)

def train():
    if not os.path.exists('saves'): os.makedirs('saves')
    
    env = CarEnv_time()
    model = CarNetwork()
    
    # Lista torów do nauki
    tracks_list = ["Race_Track3", "Race_Track4", "Race_Track5", "Race_Track6"]
    stats = load_stats()
    
    # Inicjalizacja brakujących torów w JSON
    for t in tracks_list:
        if t not in stats:
            stats[t] = {"global_best_lap": float('inf')}

    # WCZYTYWANIE POPRZEDNIEGO MISTRZA (Twoje 20s)
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
        print("Wczytano bazowy model. Rozpoczynam Multi-Track Fine-tuning")

    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Niższy LR dla stabilności

    for epoch in range(1, 5001):
        # 1. Losujemy tor dla tej epoki
        current_track = random.choice(tracks_list)
        env.track.load_track(current_track)
        env.best_lap = stats[current_track]["global_best_lap"]
        
        state = env.reset()
        log_probs, rewards = [], []
        done = False
        epoch_reward = 0
        start_tick = pygame.time.get_ticks()

        while not done:
            state_t = torch.from_numpy(state).float()
            logits = model(state_t)
            probs = torch.softmax(logits, dim=0)
            dist = Categorical(probs)
            
            # Mniejszy epsilon, bo model już dużo umie
            epsilon = max(0.02, 0.1 - (epoch / 3000))
            if random.random() < epsilon:
                action = torch.tensor(random.randint(0, 3))
            else:
                action = dist.sample()

            log_probs.append(dist.log_prob(action))
            state, reward, done = env.step(action.item())
            rewards.append(reward)
            epoch_reward += reward

            env.render(epoch=epoch, track_name=current_track)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return

        # 2. Obliczanie zysku (Reinforcement Learning)
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.98 * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        policy_loss = [-lp * r for lp, r in zip(log_probs, discounted_rewards)]
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        # 3. Sprawdzanie rekordu dla AKTUALNEGO toru
        if env.best_lap < stats[current_track]["global_best_lap"]:
            stats[current_track]["global_best_lap"] = env.best_lap
            save_stats(stats)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"!!! NOWY REKORD na {current_track}: {env.best_lap:.2f}s !!!")

        # 4. Logowanie co 10 epok
        if epoch % 2 == 0:
            lap_time = (pygame.time.get_ticks() - start_tick) / 1000.0
            msg = f"Ep: {epoch} | Track: {current_track} | Time: {lap_time:.2f}s | Best: {env.best_lap:.2f}s"
            print(msg)
            logging_method.log('training_multi.txt', msg)

if __name__ == "__main__":
    train()