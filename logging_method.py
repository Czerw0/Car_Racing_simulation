import os 
from datetime import datetime
import time

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

lap_times = []
best_laps = []
epochs_nums = []

def log(filename, message):
    os.makedirs('logs', exist_ok=True)
    with open(f'logs/{filename}', 'a', encoding='utf-8') as f:
        f.write(f'[{timestamp}] {message}\n')

def save_for_plotting(lap_time, best_lap,epoch):
    lap_times.append(lap_time)
    best_laps.append(best_lap)
    epochs_nums.append(epoch)