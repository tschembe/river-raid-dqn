"""
Deep Q-Network (DQN) Reinforcement Learning Agent in Pure Python + NumPy.
Features:
- 3-Layer Multi-Layer Perceptron (13 -> 64 -> 64 -> 5) with LeakyReLU
- Experience Replay Buffer
- Separate Target Network for stable Bellman updates
- Vectorized Mini-Batch Backpropagation with Gradient Clipping
- Save / Load model checkpoint (.npz)
"""

from __future__ import annotations

import os
from collections import deque
import random
from typing import Deque, List, Tuple
import numpy as np

ACTION_NAMES = ["COAST", "LEFT <-", "RIGHT ->", "FAST ^", "SLOW v"]


class MLP:
    """3-Layer Neural Network implemented in vectorized NumPy."""

    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, output_dim: int = 5) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)

        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, hidden_dim), dtype=np.float32)

        self.W3 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros((1, output_dim), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass supporting both single state (1D) and batch (2D)."""
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.where(self.z1 > 0, self.z1, 0.01 * self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.where(self.z2 > 0, self.z2, 0.01 * self.z2)

        self.out = self.a2 @ self.W3 + self.b3
        return self.out[0] if single else self.out

    def train_batch(self, states: np.ndarray, target_q: np.ndarray, lr: float) -> float:
        """Vectorized mini-batch gradient descent step."""
        batch_size = states.shape[0]
        pred_q = self.forward(states)

        diff = pred_q - target_q
        loss = float(np.mean(0.5 * (diff ** 2)))

        # Clip output error gradient (Huber-like stability)
        d_out = np.clip(diff, -2.0, 2.0) / batch_size

        # Layer 3 gradients
        dW3 = self.a2.T @ d_out
        db3 = np.sum(d_out, axis=0, keepdims=True)

        # Layer 2 gradients
        da2 = d_out @ self.W3.T
        dz2 = da2 * np.where(self.z2 > 0, 1.0, 0.01)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Layer 1 gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * np.where(self.z1 > 0, 1.0, 0.01)
        dW1 = self.x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Parameter updates
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        return loss

    def copy_from(self, other: "MLP") -> None:
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


class RiverRaidDQNAgent:
    """Deep Q-Network Agent with Experience Replay and Target Network."""

    def __init__(
        self,
        state_dim: int = 13,
        action_dim: int = 5,
        hidden_dim: int = 64,
        model_path: str = "river_raid_dqn.npz",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model_path = model_path

        self.policy_net = MLP(state_dim, hidden_dim, action_dim)
        self.target_net = MLP(state_dim, hidden_dim, action_dim)
        self.target_net.copy_from(self.policy_net)

        # Hyperparameters
        self.gamma = 0.96
        self.lr = 0.003
        self.epsilon = 0.35
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9993
        self.batch_size = 64

        # Experience Replay Buffer
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=20000)

        # Telemetry
        self.episode = 1
        self.total_steps = 0
        self.episode_reward = 0.0
        self.best_score = 0
        self.last_action = 0
        self.last_q_values = np.zeros(action_dim, dtype=np.float32)
        self.last_loss = 0.0

        self.load()

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        q_values = self.policy_net.forward(state)
        self.last_q_values = q_values

        if explore and random.random() < self.epsilon:
            self.last_action = random.randrange(self.action_dim)
            return self.last_action

        self.last_action = int(np.argmax(q_values))
        return self.last_action

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))
        self.episode_reward += reward
        self.total_steps += 1

    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        current_q = self.policy_net.forward(states).copy()
        next_q = self.target_net.forward(next_states)
        max_next_q = np.max(next_q, axis=1)

        targets = rewards + (1.0 - dones) * self.gamma * max_next_q
        current_q[np.arange(self.batch_size), actions] = targets

        self.last_loss = self.policy_net.train_batch(states, current_q, self.lr)

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.total_steps % 200 == 0:
            self.target_net.copy_from(self.policy_net)

    def on_episode_end(self, score: int) -> None:
        if score > self.best_score:
            self.best_score = score
        self.episode += 1
        self.episode_reward = 0.0
        self.target_net.copy_from(self.policy_net)
        if self.episode % 5 == 0:
            self.save()

    def save(self) -> None:
        np.savez(
            self.model_path,
            W1=self.policy_net.W1,
            b1=self.policy_net.b1,
            W2=self.policy_net.W2,
            b2=self.policy_net.b2,
            W3=self.policy_net.W3,
            b3=self.policy_net.b3,
            episode=np.array([self.episode], dtype=np.int32),
            epsilon=np.array([self.epsilon], dtype=np.float32),
            best_score=np.array([self.best_score], dtype=np.int32),
        )

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        try:
            data = np.load(self.model_path)
            self.policy_net.W1 = data["W1"]
            self.policy_net.b1 = data["b1"]
            self.policy_net.W2 = data["W2"]
            self.policy_net.b2 = data["b2"]
            self.policy_net.W3 = data["W3"]
            self.policy_net.b3 = data["b3"]
            self.target_net.copy_from(self.policy_net)
            self.episode = int(data["episode"][0])
            self.epsilon = float(data["epsilon"][0])
            self.best_score = int(data["best_score"][0])
            return True
        except Exception:
            return False

    def reset_brain(self) -> None:
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        self.policy_net = MLP(self.state_dim, self.hidden_dim, self.action_dim)
        self.target_net = MLP(self.state_dim, self.hidden_dim, self.action_dim)
        self.target_net.copy_from(self.policy_net)
        self.memory.clear()
        self.episode = 1
        self.epsilon = 0.40
        self.episode_reward = 0.0
        self.best_score = 0
