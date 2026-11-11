# River Raid — Double DQN Reinforcement Learning

A River Raid simulation environment and Double Deep Q-Network (DQN) autopilot written in Python + NumPy, paired with a clean engineering web visualizer.

## Features
- **Procedural River Canyon Environment** (`river_raid_env.py`): Winding river channels, split islands, and crossing jets with 7-ray Lidar sensor casting (`-75°` to `+75°`) and lookahead channel center tracking. Seedable (`RiverRaidEnv(seed=...)`, `reset(seed=...)`) and fast (~8.7k steps/s, O(1) terrain lookup).
- **Double DQN Agent** (`rl_agent.py`): 3-layer MLP (`13 -> 64 -> 64 -> 5`) with LeakyReLU, Adam, Huber loss on the taken action, Double-DQN targets, a 50,000-transition NumPy replay ring buffer, target sync every 1,000 gradient steps, and checkpointing (`river_raid_dqn.npz`).
- **Engineering Telemetry GUI** (`index.html`, `style.css`, `game.js`, `rl_agent.js`): Lidar vector overlays, live Q-value bars, 13D state vector inspector, and up to `40x` simulation warp. The browser env mirrors the Python physics exactly, so Python-trained weights behave the same in both.

## Quick Start

### 1. Train the RL Agent in Python (Headless CLI)
```bash
python3 river_raid.py --train 300               # continue training the saved checkpoint
python3 river_raid.py --train 300 --fresh --seed 1 --eval 30   # from scratch, then benchmark
```

### 2. Benchmark a Policy
```bash
python3 river_raid.py --eval 30                  # greedy, 30 fixed-seed rivers
```
Every checkpoint is scored on the same 30 river layouts, so numbers are directly comparable.

### 3. Launch the Engineering Web Visualizer
```bash
python3 river_raid.py --serve                    # optional: --port 8001
```
Then open `http://127.0.0.1:8000`. When the Python brain loads, the page switches to greedy playback (**Online Learning: OFF**); toggle it back on to keep training in the browser.

### 4. Play or Watch in the Terminal (Curses UI)
```bash
python3 river_raid.py --terminal
```

### Tests
```bash
python3 -m unittest discover tests
```
Includes a parity test that fails if `game.js` physics drift from `river_raid_env.py`.
