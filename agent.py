import torch.nn as nn
import torch
from collections import deque
import random
import os
from constants import ACTIONS

SAVE_PATH = os.path.join(os.path.dirname(__file__), "pacman_agent.pth")

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
    
    def forward(self, x):
        return self.net(x)
        
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self):
        self.net = QNet()
        self.buffer = ReplayBuffer(5000)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.steps = 0
        self.action_steps = 0
        if os.path.exists(SAVE_PATH):
            checkpoint = torch.load(SAVE_PATH)
            self.net.load_state_dict(checkpoint["model"])
            self.epsilon = checkpoint["epsilon"]
            self.steps = checkpoint["steps"]
            self.action_steps = checkpoint.get("action_steps", 0)
            print(f"Loaded agent from {SAVE_PATH} (train_steps={self.steps}, action_steps={self.action_steps}, epsilon={self.epsilon:.5f})")

    def train(self):
        if len(self.buffer) < 64:
            return

        self.steps += 1
        if self.steps % 1000 == 0:
            torch.save({"model": self.net.state_dict(), "epsilon": self.epsilon, "steps": self.steps, "action_steps": self.action_steps}, SAVE_PATH)

        states, actions, rewards, next_states, dones = self.buffer.sample(64)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # gather the Q-value for each sample's chosen action
        prediction = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # max Q-value per sample, and stop gradient through target network
        with torch.no_grad():
            target = rewards + self.gamma * self.net(next_states).max(dim=1).values * (1 - dones)
        loss = nn.functional.mse_loss(prediction, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        x = torch.tensor(state, dtype=torch.float32)
        self.action_steps += 1
        self.epsilon = max(0.05, self.epsilon * 0.999)
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        with torch.no_grad():
            return ACTIONS[self.net.forward(x).argmax().item()]
