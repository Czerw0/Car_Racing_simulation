import torch
import torch.nn as nn
import torch.nn.functional as F

class CarNetwork(nn.Module):
    # 6 input points - 5 sensors and speed
    def __init__(self, input_size=6, output_size=4):
        super(CarNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x