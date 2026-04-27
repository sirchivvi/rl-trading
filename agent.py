import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------------------------------------------------
# Q-Network
# -----------------------------------------------------------------------
class QNetwork(nn.Module):
    """
    A small fully-connected neural network that estimates Q-values.

    Input  : state vector  (state_size,)
    Output : Q-values for each action  (action_size,)
    """

    def __init__(self, state_size: int, action_size: int):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------------------
class ReplayBuffer:
    """
    A simple circular replay buffer that stores experience tuples.
    (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# -----------------------------------------------------------------------
# DQN Agent
# -----------------------------------------------------------------------
class DQNAgent:
    """
    Deep Q-Network agent with ε-greedy exploration and experience replay.

    Hyperparameters are intentionally simple and commented so a student
    can understand what each one does.
    """

    def __init__(
        self,
        state_size:    int,
        action_size:   int,
        lr:            float = 1e-3,    # learning rate
        gamma:         float = 0.9,     # discount factor (finite horizon)
        epsilon:       float = 1.0,     # starting exploration rate
        epsilon_min:   float = 0.05,    # minimum exploration rate
        epsilon_decay: float = 0.995,   # decay per episode
        batch_size:    int   = 64,      # mini-batch size for training
        buffer_size:   int   = 10_000,  # replay buffer capacity
    ):
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size

        # Two networks: online (trained) and target (stable reference)
        self.online_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self._sync_target()                           # copy weights

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()
        self.memory    = ReplayBuffer(buffer_size)

    # ------------------------------------------------------------------
    # Target network helpers
    # ------------------------------------------------------------------
    def _sync_target(self):
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)   # explore

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax().item())             # exploit

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------
    def learn(self) -> float:
        """
        Sample a mini-batch from replay buffer and update the online network.
        Returns the loss value for logging.
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions).unsqueeze(1)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q-values for taken actions
        current_q = self.online_net(states_t).gather(1, actions_t)

        # Target Q-values (Bellman equation)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q   = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------
    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str):
        self.online_net.load_state_dict(torch.load(path, map_location="cpu"))
        self._sync_target()