import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.learner.common.Hyperparameters import *

import random
class Qnet(nn.Module):  # Qnet
    def __init__(self, input_layer, output_layer):
        super(Qnet, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.fc1 = nn.Linear(self.input_layer, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, self.output_layer)

    def forward(self, x):
        if (Hyperparameters.Q_net_activation_function == "ReLU"):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        elif (Hyperparameters.Q_net_activation_function == "SELU"):
            x = F.selu(self.fc1(x))
            x = F.selu(self.fc2(x))
            x = F.selu(self.fc3(x))

        elif (Hyperparameters.Q_net_activation_function == "Swish"):
            x = self.fc1(x) * torch.sigmoid(self.fc1(x))
            x = self.fc2(x) * torch.sigmoid(self.fc2(x))
            x = self.fc3(x) * torch.sigmoid(self.fc3(x))

        elif (Hyperparameters.Q_net_activation_function == "tanh"):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))

        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        out = self.forward(obs.unsqueeze(0))
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.output_layer - 1)
        else:
            return out.argmax().item()

    def sample_action_list(self, obs, epsilon):
        out = self.forward(obs)
        return out

    def select_action(self, obs, epsilon):
        out = self.forward(obs)
        return out.argmax().item(), out

    def action_masking_action(self, obs, epsilon):
        return
