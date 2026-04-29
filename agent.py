import torch.nn as nn
import torch
from collections import deque
import random
import os
from constants import ACTIONS

SAVE_PATH = os.path.join(os.path.dirname(__file__), "pacman_agent.pth")

# The neural network that learns to play Pac-Man.
# Input: 30 numbers describing the game state (positions, ghosts, pellets, etc.)
# Hidden layer: 128 neurons — enough capacity to learn patterns without being too slow
# Output: 5 numbers, one Q-value per possible action (STOP, UP, DOWN, LEFT, RIGHT)
# The action with the highest Q-value is considered "best" by the network.
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

# The replay buffer stores past game moments (state, action, reward, next_state, done).
# Instead of training on the most recent frame only, we sample randomly from this history.
# This breaks the correlation between consecutive frames, which stabilises training —
# without it the network would just overfit to whatever just happened.
# Capacity 50000: large enough to hold diverse experience across many games.
# Old memories are automatically dropped when the buffer is full (deque with maxlen).
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

        # 50000 memories — large buffer means the agent learns from a wide variety
        # of situations, not just the most recent ones.
        self.buffer = ReplayBuffer(50000)

        # Adam optimiser with a standard learning rate for DQN.
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        # Discount factor: how much future rewards matter vs immediate ones.
        # 0.99 means the agent cares a lot about the future (almost as much as now).
        self.gamma = 0.99

        # Exploration rate. Starts at 1.0 (fully random) and decays toward 0.05
        # (mostly greedy, but 5% random to avoid getting completely stuck).
        self.epsilon = 1.0

        self.steps = 0        # counts gradient update steps (for checkpoint timing)
        self.action_steps = 0 # counts how many times the agent picked an action
        self.scores = []      # final score of every game, used for plotting progress

        # Resume from a previous training session if a checkpoint exists.
        if os.path.exists(SAVE_PATH):
            checkpoint = torch.load(SAVE_PATH)
            self.net.load_state_dict(checkpoint["model"])
            self.epsilon = checkpoint["epsilon"]
            self.steps = checkpoint["steps"]
            self.action_steps = checkpoint.get("action_steps", 0)
            self.scores = checkpoint.get("scores", [])
            print(f"Loaded agent from {SAVE_PATH} (train_steps={self.steps}, action_steps={self.action_steps}, epsilon={self.epsilon:.5f})")

    def on_game_end(self):
        # Decay epsilon once per game, not once per frame.
        # The old code decayed per action step, which meant epsilon hit its minimum
        # within the very first game — the agent never actually explored.
        # At 0.97 per game, exploration stays meaningful for ~100 games before
        # settling at the 0.05 minimum.
        self.epsilon = max(0.05, self.epsilon * 0.97)

    def train(self):
        # Don't train until there are enough memories to sample a full batch.
        if len(self.buffer) < 64:
            return

        self.steps += 1
        # Save a checkpoint every 1000 training steps so progress isn't lost.
        if self.steps % 1000 == 0:
            torch.save({
                "model": self.net.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "action_steps": self.action_steps,
                "scores": self.scores
            }, SAVE_PATH)

        states, actions, rewards, next_states, dones = self.buffer.sample(64)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # What Q-value did the network predict for the action that was actually taken?
        prediction = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # What should the Q-value have been?
        # = immediate reward + discounted best future value (Bellman equation).
        # torch.no_grad() stops gradients flowing into the target calculation —
        # if the target moved every step it would be a "chasing your own tail" problem.
        # (A proper target network would be even more stable, but this is simpler.)
        with torch.no_grad():
            target = rewards + self.gamma * self.net(next_states).max(dim=1).values * (1 - dones)

        # MSE loss: push predictions toward targets.
        loss = nn.functional.mse_loss(prediction, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        x = torch.tensor(state, dtype=torch.float32)
        self.action_steps += 1

        # Epsilon-greedy: with probability epsilon pick a random action (explore),
        # otherwise pick the action the network thinks is best (exploit).
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        with torch.no_grad():
            return ACTIONS[self.net.forward(x).argmax().item()]
