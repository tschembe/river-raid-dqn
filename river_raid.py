#!/usr/bin/env python3
"""
River Raid - Complete Python Application
Includes:
1. Full Python + NumPy River Raid Environment (`river_raid_env.py`)
2. Deep Q-Network (DQN) Reinforcement Learning Agent (`rl_agent.py`)
3. Interactive Terminal Curses Arcade UI (`--terminal`)
4. Headless Fast RL Training CLI (`--train <episodes>`)
5. Greedy policy benchmark (`--eval <episodes>`)
6. Browser Visualizer Server (`--serve`)
"""

from __future__ import annotations

import argparse
import curses
import functools
import json
import os
import sys
import random
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

import numpy as np

from river_raid_env import RiverRaidEnv
from rl_agent import ACTION_NAMES, DEFAULT_MODEL_PATH, RiverRaidDQNAgent

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Mode 1: Fast Headless CLI Training in Python
# ---------------------------------------------------------------------------
def train_headless(
    episodes: int = 100,
    max_steps: int = 4000,
    model_path: str = DEFAULT_MODEL_PATH,
    fresh: bool = False,
    seed: int | None = None,
) -> RiverRaidDQNAgent:
    if fresh and os.path.exists(model_path):
        os.remove(model_path)
    env = RiverRaidEnv(seed=seed)
    agent = RiverRaidDQNAgent(model_path=model_path)

    print("=" * 68)
    print(f"🚀 RIVER RAID - PYTHON DQN REINFORCEMENT LEARNING TRAINER")
    print(f"Architecture : Double DQN, MLP (13 -> 64 -> 64 -> 5) | Adam | Replay Buffer: {agent.memory.capacity:,}")
    print(f"Starting Ep  : {agent.episode} | Epsilon: {agent.epsilon:.3f} | Best Score: {agent.best_score}")
    print("=" * 68)

    scores = []
    start_time = time.time()

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        while steps < max_steps:
            action = agent.act(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            agent.replay()

            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break

        scores.append(env.score)
        agent.on_episode_end(env.score)
        avg_recent = float(np.mean(scores[-20:]))

        if ep % 5 == 0 or ep == 1 or ep == episodes:
            elapsed = time.time() - start_time
            print(
                f"Episode {agent.episode - 1:4d} | "
                f"Score: {env.score:5d} | "
                f"Avg(20): {avg_recent:6.1f} | "
                f"Best: {agent.best_score:5d} | "
                f"Reward: {total_reward:7.1f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Steps: {steps:4d} | "
                f"{elapsed:5.1f}s"
            )

    agent.save()
    print("=" * 68)
    print(f"Throughput   : {agent.total_steps / (time.time() - start_time):,.0f} steps/s")
    print(f"✅ Training complete! Saved trained model to '{agent.model_path}' (Best Score: {agent.best_score})")
    print("=" * 68)
    return agent


# ---------------------------------------------------------------------------
# Mode 1b: Greedy (no exploration) benchmark of a saved policy
# ---------------------------------------------------------------------------
def evaluate(
    episodes: int = 20,
    max_steps: int = 4000,
    model_path: str = DEFAULT_MODEL_PATH,
    seed: int = 0,
) -> Dict[str, float]:
    env = RiverRaidEnv()
    agent = RiverRaidDQNAgent(model_path=model_path)
    scores, sectors, crashes = [], [], 0
    for ep in range(episodes):
        # Episode i always sees the same river layout, so checkpoints are comparable
        state = env.reset(seed=seed + ep)
        for _ in range(max_steps):
            state, _, done, _ = env.step(agent.act(state, explore=False))
            if done:
                crashes += 1
                break
        scores.append(env.score)
        sectors.append(env.sector)

    result = {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean_sector": float(np.mean(sectors)),
        "crash_rate": crashes / episodes,
    }
    print(f"Greedy eval of '{model_path}' over {episodes} seeded episodes (cap {max_steps} steps):")
    print(
        f"  score mean {result['mean']:.0f} | median {result['median']:.0f} | "
        f"min {result['min']:.0f} | max {result['max']:.0f}"
    )
    print(f"  mean sector reached {result['mean_sector']:.2f} | crash rate {result['crash_rate']:.0%}")
    return result


# ---------------------------------------------------------------------------
# Mode 2: Native Terminal Curses Arcade Game + Live RL Autopilot
# ---------------------------------------------------------------------------
def run_curses_game(stdscr: Any) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(30)

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)   # Water
    curses.init_pair(2, curses.COLOR_GREEN, -1)                  # Land
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLUE)  # Player / Score
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLUE)     # Enemy Jet
    curses.init_pair(5, curses.COLOR_CYAN, -1)                   # HUD / RL

    env = RiverRaidEnv()
    agent = RiverRaidDQNAgent()
    state = env.reset()
    rl_enabled = True
    warp = 1

    while True:
        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            agent.save()
            break
        elif key in (ord("t"), ord("T")):
            rl_enabled = not rl_enabled
        elif key in (ord("w"), ord("W")):
            warp = 8 if warp == 1 else 1
        elif key in (ord("r"), ord("R")):
            agent.reset_brain()
            state = env.reset()

        manual_action = 0
        if key == curses.KEY_LEFT:
            manual_action = 1
        elif key == curses.KEY_RIGHT:
            manual_action = 2
        elif key == curses.KEY_UP:
            manual_action = 3
        elif key == curses.KEY_DOWN:
            manual_action = 4

        steps = warp if rl_enabled else 1
        for _ in range(steps):
            action = agent.act(state, explore=True) if rl_enabled else manual_action
            next_state, reward, done, _ = env.step(action)
            if rl_enabled:
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
            state = next_state
            if done:
                if rl_enabled:
                    agent.on_episode_end(env.score)
                state = env.reset()

        # Render to terminal
        max_y, max_x = stdscr.getmaxyx()
        view_h = min(max_y - 5, 28)
        view_w = min(max_x - 2, 72)
        if view_h < 10 or view_w < 30:
            stdscr.erase()
            stdscr.addstr(0, 0, "Please enlarge terminal window!")
            stdscr.refresh()
            continue

        stdscr.erase()
        mode_str = f"🤖 RL AUTOPILOT ({warp}x)" if rl_enabled else "🎮 MANUAL PILOT"
        header = (
            f" RIVER RAID PYTHON | {mode_str} | Score: {env.score:05d} | "
            f"Best: {agent.best_score:05d} | Ep: {agent.episode} | Eps: {agent.epsilon:.2f} "
        )
        stdscr.addstr(0, 0, header[: max_x - 1], curses.color_pair(5) | curses.A_BOLD)

        for row in range(view_h):
            world_y = (row / view_h) * env.PLAY_H
            s = env.get_slice_at_y(world_y)
            left_b = int(((s.center - s.width / 2.0) / env.W) * view_w)
            right_b = int(((s.center + s.width / 2.0) / env.W) * view_w)
            isl_l = int(((s.center - s.island_width / 2.0) / env.W) * view_w) if s.island_width > 6 else -1
            isl_r = int(((s.center + s.island_width / 2.0) / env.W) * view_w) if s.island_width > 6 else -1

            for col in range(view_w):
                if col < left_b or col > right_b or (isl_l <= col <= isl_r and s.island_width > 6):
                    stdscr.addstr(row + 1, col, "█", curses.color_pair(2))
                else:
                    stdscr.addstr(row + 1, col, " ", curses.color_pair(1))

        # Draw crossing jets
        for j in env.jets:
            jr = int((j.y / env.PLAY_H) * view_h) + 1
            jc = int((j.x / env.W) * view_w)
            if 1 <= jr <= view_h and 1 <= jc < view_w - 1:
                stdscr.addstr(jr, jc - 1, "=✈=", curses.color_pair(4) | curses.A_BOLD)

        # Draw player plane
        pr = int((env.player_y / env.PLAY_H) * view_h) + 1
        pc = int((env.player_x / env.W) * view_w)
        if 1 <= pr <= view_h and 1 <= pc < view_w - 1:
            stdscr.addstr(pr, pc - 1, "/▲\\", curses.color_pair(3) | curses.A_BOLD)

        footer = " [T] Toggle RL/Manual | [W] Warp 1x/8x | [Arrows] Steer | [R] Reset Brain | [Q] Quit & Save"
        stdscr.addstr(view_h + 2, 0, footer[: max_x - 1], curses.color_pair(5))
        stdscr.refresh()


# ---------------------------------------------------------------------------
# Mode 3: Python Backend Server syncing Python DQN weights with Browser UI
# ---------------------------------------------------------------------------
class PythonRLRequestHandler(SimpleHTTPRequestHandler):
    agent: RiverRaidDQNAgent = None  # type: ignore
    env_config: Dict[str, float] = {}

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Keep terminal output clean

    def _send_json(self, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/api/config":
            self._send_json(self.env_config)
            return
        if self.path == "/api/brain":
            payload = {
                "episode": self.agent.episode,
                "epsilon": self.agent.epsilon,
                "bestScore": self.agent.best_score,
                "weights": {
                    "l1w": self.agent.policy_net.W1.T.flatten().tolist(),
                    "l1b": self.agent.policy_net.b1.flatten().tolist(),
                    "l2w": self.agent.policy_net.W2.T.flatten().tolist(),
                    "l2b": self.agent.policy_net.b2.flatten().tolist(),
                    "l3w": self.agent.policy_net.W3.T.flatten().tolist(),
                    "l3b": self.agent.policy_net.b3.flatten().tolist(),
                },
            }
            self._send_json(payload)
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="River Raid + Deep Q-Network RL in Python")
    parser.add_argument(
        "--train",
        type=int,
        default=0,
        metavar="EPISODES",
        help="Train the Python DQN agent headlessly for N episodes",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Play / watch the RL agent live inside the terminal using curses",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start local HTTP server on port 8000 with Python-trained weights",
    )
    parser.add_argument(
        "--eval",
        type=int,
        default=0,
        metavar="EPISODES",
        help="Benchmark the saved policy greedily over N seeded episodes",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Checkpoint path (.npz)")
    parser.add_argument("--fresh", action="store_true", help="Discard the checkpoint and train from scratch")
    parser.add_argument("--seed", type=int, default=None, help="Seed Python/NumPy RNGs for reproducible runs")
    parser.add_argument("--port", type=int, default=8000, help="Port for --serve")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.train > 0:
        train_headless(args.train, model_path=args.model, fresh=args.fresh, seed=args.seed)
        if args.eval > 0:
            evaluate(args.eval, model_path=args.model)
        return

    if args.eval > 0:
        evaluate(args.eval, model_path=args.model)
        return

    if args.terminal:
        curses.wrapper(run_curses_game)
        return

    # Default: run a quick pre-training warmup if no checkpoint exists, then serve or summarize
    if not os.path.exists(args.model):
        print("No saved Python model found — pre-training DQN agent for 60 episodes in Python...")
        agent = train_headless(60, model_path=args.model)
    else:
        agent = RiverRaidDQNAgent(model_path=args.model)
        print(f"Loaded trained Python DQN model '{agent.model_path}' (Episode {agent.episode}, Best Score: {agent.best_score})")

    if args.serve:
        PythonRLRequestHandler.agent = agent
        PythonRLRequestHandler.env_config = RiverRaidEnv().config()
        port = args.port
        handler = functools.partial(PythonRLRequestHandler, directory=ROOT_DIR)
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        print(f"🌐 Serving River Raid at http://127.0.0.1:{port} (Press Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()
