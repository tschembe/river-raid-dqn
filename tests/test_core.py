"""Core checks for the env, the DQN agent and browser/Python physics parity.

Run with: python3 -m unittest discover tests
"""

from __future__ import annotations

import os
import random
import re
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from river_raid_env import RiverRaidEnv  # noqa: E402
from rl_agent import MLP, RiverRaidDQNAgent  # noqa: E402


def linear_slice_lookup(env: RiverRaidEnv, y: float):
    best = env.slices[0]
    for s in env.slices:
        if abs(s.y - y) < abs(best.y - y):
            best = s
    return best


class EnvTests(unittest.TestCase):
    def test_indexed_slice_lookup_matches_linear_scan(self) -> None:
        env = RiverRaidEnv(seed=3)
        rng = random.Random(0)
        for _ in range(300):
            env.step(rng.randrange(5))
            if env.done:
                env.reset()
            for _ in range(10):
                y = rng.uniform(-20.0, env.PLAY_H + 20.0)
                self.assertIs(env.get_slice_at_y(y), linear_slice_lookup(env, y))

    def test_same_seed_same_episode(self) -> None:
        actions = [random.Random(1).randrange(5) for _ in range(500)]
        runs = []
        for _ in range(2):
            env = RiverRaidEnv()
            obs = [env.reset(seed=42)]
            for a in actions:
                o, _, done, _ = env.step(a)
                obs.append(o)
                if done:
                    break
            runs.append((env.score, np.array(obs)))
        self.assertEqual(runs[0][0], runs[1][0])
        np.testing.assert_array_equal(runs[0][1], runs[1][1])

    def test_observation_shape_and_range(self) -> None:
        env = RiverRaidEnv(seed=0)
        obs = env.reset()
        self.assertEqual(obs.shape, (13,))
        self.assertTrue(np.all((obs[:7] > 0.0) & (obs[:7] <= 1.0)))


class AgentTests(unittest.TestCase):
    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "brain.npz")
            a = RiverRaidDQNAgent(model_path=path)
            a.episode, a.best_score = 17, 4321
            a.save()
            b = RiverRaidDQNAgent(model_path=path)
            self.assertEqual((b.episode, b.best_score), (17, 4321))
            np.testing.assert_array_equal(a.policy_net.W2, b.policy_net.W2)
            state = np.random.rand(13).astype(np.float32)
            np.testing.assert_allclose(a.policy_net.forward(state), b.policy_net.forward(state))

    def test_train_batch_fits_fixed_targets(self) -> None:
        np.random.seed(0)
        net = MLP()
        states = np.random.rand(64, 13).astype(np.float32)
        actions = np.random.randint(0, 5, size=64)
        targets = np.random.randn(64).astype(np.float32)
        first = net.train_batch(states, actions, targets, lr=1e-3)
        for _ in range(500):
            last = net.train_batch(states, actions, targets, lr=1e-3)
        self.assertLess(last, first * 0.2)


class ParityTests(unittest.TestCase):
    """The browser re-implements the env; Python-trained weights only transfer if physics match."""

    def test_game_js_constants_match_env(self) -> None:
        with open(os.path.join(ROOT, "game.js")) as f:
            js = f.read()
        with open(os.path.join(ROOT, "index.html")) as f:
            html = f.read()
        cfg = RiverRaidEnv().config()

        def js_num(pattern: str) -> float:
            m = re.search(pattern, js)
            self.assertIsNotNone(m, pattern)
            return float(m.group(1))

        canvas = re.search(r'id="gameCanvas" width="(\d+)" height="(\d+)"', html)
        self.assertEqual((float(canvas.group(1)), float(canvas.group(2))), (cfg["W"], cfg["PLAY_H"]))
        self.assertEqual(js_num(r"const SLICE_H = ([\d.]+);"), cfg["SLICE_H"])
        self.assertEqual(cfg["PLAY_H"] - js_num(r"const START_Y = PLAY_H - ([\d.]+);"), cfg["playerY"])
        self.assertEqual(js_num(r"const STEER_STEP = ([\d.]+);"), cfg["steerStep"])
        self.assertEqual(js_num(r"const RAY_STEP = ([\d.]+);"), cfg["rayStep"])
        self.assertEqual(js_num(r"const RAY_MAX = ([\d.]+);"), cfg["rayMax"])
        self.assertEqual(js_num(r"\n    w: ([\d.]+),"), cfg["playerW"])
        self.assertEqual(js_num(r"\n    h: ([\d.]+),"), cfg["playerH"])
        self.assertEqual(js_num(r"minSpeed: ([\d.]+),"), cfg["minSpeed"])
        self.assertEqual(js_num(r"maxSpeed: ([\d.]+),"), cfg["maxSpeed"])
        self.assertEqual(js_num(r"cruiseSpeed: ([\d.]+),"), cfg["cruiseSpeed"])


if __name__ == "__main__":
    unittest.main()
