# River Raid — Deep Q-Network (DQN) Reinforcement Learning

A River Raid simulation environment and Deep Q-Network (DQN) Reinforcement Learning autopilot written in Python + NumPy, paired with a clean engineering web visualizer.

## Features
- **Procedural River Canyon Environment** (`river_raid_env.py`): Winding river channels, split islands, and crossing jets with 7-ray Lidar sensor casting (`-75°` to `+75°`) and lookahead channel center tracking.
- **Deep Q-Network (DQN) Agent** (`rl_agent.py`, `rl_agent.js`): 3-layer Multi-Layer Perceptron (`13 -> 64 -> 64 -> 5`) with LeakyReLU activations, Experience Replay (`20,000` transitions), target network updates, and checkpoint saving (`river_raid_dqn.npz`).
- **Engineering Telemetry GUI** (`index.html`, `style.css`, `game.js`): Clean viewport with Lidar vector overlays, live Q-value bar graphs, 13D state vector inspector, and up to `40x` simulation warp speed.

## Quick Start

### 1. Train the RL Agent in Python (Headless CLI)
```bash
python3 river_raid.py --train 100
```

### 2. Launch the Engineering Web Visualizer
```bash
python3 river_raid.py --serve
```
Then open `http://127.0.0.1:8000` in your browser.

### 3. Play or Watch in the Terminal (Curses UI)
```bash
python3 river_raid.py --terminal
```
