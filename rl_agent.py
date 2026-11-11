"""
Deep Q-Network (DQN) Reinforcement Learning Agent in Pure Python + NumPy.
Features:
- 3-Layer Multi-Layer Perceptron (13 -> 64 -> 64 -> 5) with LeakyReLU
- Adam optimizer + Huber loss on the taken action only
- Double DQN targets (policy net picks the action, target net scores it)
- NumPy ring-buffer Experience Replay
- Separate Target Network synced every N gradient steps
- Save / Load model checkpoint (.npz)
"""

from __future__ import annotations

import os
import random
from typing import Optional
import numpy as np

ACTION_NAMES = ["COAST", "LEFT <-", "RIGHT ->", "FAST ^", "SLOW v"]

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "river_raid_dqn.npz")

PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")


class MLP:
    """3-Layer Neural Network implemented in vectorized NumPy, trained with Adam."""

    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, output_dim: int = 5) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # He initialization
        self.W1 = (np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)

        self.W2 = (np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros((1, hidden_dim), dtype=np.float32)

        self.W3 = (np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b3 = np.zeros((1, output_dim), dtype=np.float32)

        self._reset_adam()

    def _reset_adam(self) -> None:
        self.adam_t = 0
        self.adam_m = {n: np.zeros_like(getattr(self, n)) for n in PARAM_NAMES}
        self.adam_v = {n: np.zeros_like(getattr(self, n)) for n in PARAM_NAMES}

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

    def train_batch(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray, lr: float) -> float:
        """One Adam step on the Huber loss between Q(s, a) and the Bellman targets."""
        batch_size = states.shape[0]
        rows = np.arange(batch_size)
        pred_q = self.forward(states)

        diff = pred_q[rows, actions] - targets
        abs_diff = np.abs(diff)
        loss = float(np.mean(np.where(abs_diff < 1.0, 0.5 * diff ** 2, abs_diff - 0.5)))

        # Huber gradient, only flowing through the action that was actually taken
        d_out = np.zeros_like(pred_q)
        d_out[rows, actions] = np.clip(diff, -1.0, 1.0) / batch_size

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

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        self._adam_update(grads, lr)
        return loss

    def _adam_update(self, grads: dict, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.adam_t += 1
        corr1 = 1.0 - beta1 ** self.adam_t
        corr2 = 1.0 - beta2 ** self.adam_t
        for name, g in grads.items():
            m = self.adam_m[name]
            v = self.adam_v[name]
            m *= beta1
            m += (1.0 - beta1) * g
            v *= beta2
            v += (1.0 - beta2) * (g * g)
            param = getattr(self, name)
            param -= (lr * (m / corr1) / (np.sqrt(v / corr2) + eps)).astype(np.float32)

    def copy_from(self, other: "MLP") -> None:
        for name in PARAM_NAMES:
            setattr(self, name, getattr(other, name).copy())


class ReplayBuffer:
    """Fixed-size ring buffer of transitions stored as contiguous NumPy arrays."""

    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        i = self.idx
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        ix = np.random.randint(0, self.size, size=batch_size)
        return self.states[ix], self.actions[ix], self.rewards[ix], self.next_states[ix], self.dones[ix]

    def clear(self) -> None:
        self.idx = 0
        self.size = 0


class RiverRaidDQNAgent:
    """Double DQN Agent with Experience Replay and Target Network."""

    def __init__(
        self,
        state_dim: int = 13,
        action_dim: int = 5,
        hidden_dim: int = 64,
        model_path: Optional[str] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model_path = model_path or DEFAULT_MODEL_PATH

        self.policy_net = MLP(state_dim, hidden_dim, action_dim)
        self.target_net = MLP(state_dim, hidden_dim, action_dim)
        self.target_net.copy_from(self.policy_net)

        # Hyperparameters
        self.gamma = 0.97
        self.lr = 5e-4
        self.epsilon = 0.35
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        self.target_sync_every = 1000  # gradient steps
        self.warmup = 1000  # transitions collected before learning starts

        # Experience Replay Buffer
        self.memory = ReplayBuffer(50000, state_dim)

        # Telemetry
        self.episode = 1
        self.total_steps = 0
        self.grad_steps = 0
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
        self.memory.add(state, action, reward, next_state, done)
        self.episode_reward += reward
        self.total_steps += 1

    def replay(self) -> None:
        if len(self.memory) < max(self.batch_size, self.warmup):
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Double DQN: the online net chooses the next action, the target net evaluates it
        next_actions = np.argmax(self.policy_net.forward(next_states), axis=1)
        next_q = self.target_net.forward(next_states)[np.arange(self.batch_size), next_actions]
        targets = rewards + (1.0 - dones) * self.gamma * next_q

        self.last_loss = self.policy_net.train_batch(states, actions, targets, self.lr)
        self.grad_steps += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.grad_steps % self.target_sync_every == 0:
            self.target_net.copy_from(self.policy_net)

    def on_episode_end(self, score: int) -> None:
        if score > self.best_score:
            self.best_score = score
        self.episode += 1
        self.episode_reward = 0.0
        if self.episode % 5 == 0:
            self.save()

    def save(self, path: Optional[str] = None) -> None:
        np.savez(
            path or self.model_path,
            **{name: getattr(self.policy_net, name) for name in PARAM_NAMES},
            episode=np.array([self.episode], dtype=np.int32),
            epsilon=np.array([self.epsilon], dtype=np.float32),
            best_score=np.array([self.best_score], dtype=np.int32),
        )

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        try:
            data = np.load(self.model_path)
            for name in PARAM_NAMES:
                setattr(self.policy_net, name, data[name].astype(np.float32))
            self.policy_net._reset_adam()
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
        self.grad_steps = 0
        self.epsilon = 0.40
        self.episode_reward = 0.0
        self.best_score = 0
