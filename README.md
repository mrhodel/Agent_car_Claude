# Yahboom Raspbot V2 – Autonomous Indoor Navigation Agent

Fully autonomous adaptive navigation for the **Yahboom Raspbot V2** (Raspberry Pi 5) using only a **camera on a 2-DOF gimbal** and an **HC-SR04 ultrasonic sensor**.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                 │
│               CLI: --mode run | train | eval | test              │
├─────────────────────────────────────────────────────────────────┤
│                     Agent Orchestrator                           │
│   FSM:  INIT → EXPLORE ↔ PLAN → EXECUTE → RECOVER              │
│   • Emergency stop  • Periodic gimbal scan  • Map serialise     │
├────────────────┬──────────────────┬────────────────────────────-┤
│  Perception    │   Navigation     │         RL Agent             │
│ ─────────────  │  ──────────────  │  ─────────────────────────── │
│ VisionPipeline │ OccupancyGrid    │  PPO (Proximal Policy Opt.)  │
│ DepthEstimator │ A* Planner       │  Actor-Critic (MLP + GRU)   │
│ ObsDetector    │ MotionController │  GAE Advantage estimation   │
│ FloorDetector  │ Pure-pursuit     │  Discrete action space (11) │
│ SensorFusion   │ Recovery FSM     │  Checkpoint save/load       │
├────────────────┴──────────────────┴─────────────────────────────┤
│                  HAL  (Hardware Abstraction Layer)               │
│  MotorController (I2C Mecanum)  ·  UltrasonicSensor (HC-SR04)  │
│  Gimbal (PCA9685 pan-tilt)      ·  Camera (picamera2 / USB)    │
└─────────────────────────────────────────────────────────────────┘
```

### Key design choices

| Component | Choice | Why |
|---|---|---|
| Wheels | Mecanum (omnidirectional) | Strafe without turning; ideal for tight corridors |
| RL algorithm | PPO-clip | Stable, sample-efficient, works on-robot |
| Mapping | Probabilistic log-odds grid | Handles sensor noise; dynamic expansion |
| Path planning | A* + Gaussian safety inflation | Fast, optimal on small indoor grids |
| Depth | MiDaS-small (CPU) | <100 ms RPi 5; graceful fallback to gradient proxy |
| Detection | YOLOv5n (optional) | <6 ms on RPi 5 NPU or ~50 ms CPU |

---

## Hardware

| Component | Spec |
|---|---|
| Base | Yahboom Raspbot V2 |
| SBC | Raspberry Pi 5 (4 GB or 8 GB) |
| Drive | 4× Mecanum wheels |
| Range sensor | HC-SR04 ultrasonic (1 transducer, fixed forward on chassis) |
| Camera | USB UVC camera (primary); CSI optional |
| Gimbal | 2-DOF pan-tilt for camera only (Yahboom I2C servos) |
| Motor driver | Yahboom expansion board (I2C address `0x2B`) |

---

## Project Structure

```
Agent_car_Claude/
├── config/
│   └── robot_config.yaml      # All tuneable parameters
├── hal/
│   ├── motors.py              # Mecanum wheel controller (I2C / GPIO)
│   ├── ultrasonic.py          # HC-SR04 fixed forward sensor via I2C
│   ├── gimbal.py              # Camera pan-tilt via Yahboom I2C servos
│   └── camera.py              # USB (primary) / CSI / sim back-end
├── perception/
│   ├── vision.py              # Main pipeline (orchestrates below)
│   ├── depth_estimator.py     # MiDaS monocular depth
│   ├── obstacle_detector.py   # Depth + YOLOv5n fusion
│   ├── floor_detector.py      # HSV / IPM floor segmentation
│   └── sensor_fusion.py       # US + camera polar obstacle map
├── mapping/
│   └── occupancy_grid.py      # Log-odds 2-D grid with frontier detection
├── navigation/
│   ├── path_planner.py        # A* with diagonal moves + path smoothing
│   └── controller.py          # Pure-pursuit + discrete RL action map
├── rl/
│   ├── environment.py         # Gym-style env (robot + sim)
│   ├── policy.py              # ActorCritic (PyTorch + numpy fallback)
│   ├── agent.py               # PPO-clip + GAE + checkpoint I/O
│   ├── reward.py              # Exploration / proximity shaping
│   └── trainer.py             # Training & evaluation loops
├── agent/
│   └── orchestrator.py        # Top-level FSM tying everything together
├── models/
│   └── checkpoints/           # Saved policy weights
├── logs/                      # Runtime logs + map snapshots
├── main.py                    # CLI entry point
└── requirements.txt
```

---

## Quick Start

### 1 · Install dependencies (Raspberry Pi)

```bash
pip install -r requirements.txt --break-system-packages
```

### 2 · Hardware self-test

```bash
python main.py --mode test
```

This exercises motors, gimbal, ultrasonic, and camera in sequence.

### 3 · Train the RL policy – fast GPU training on Windows

Recommended workflow: train in simulation on a Windows PC with NVIDIA GPU
(~1 ep/s), then fine-tune with `--hardware` on the Pi.

See **[WINDOWS_TRAINING.md](WINDOWS_TRAINING.md)** for the full step-by-step guide.

Quick summary:
```powershell
# On Windows (PowerShell)
python -m venv robot_venv
.\robot_venv\Scripts\Activate.ps1
pip install -r requirements_desktop.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python main.py --mode train --episodes 1500
# ~25 min on RTX 3050  |  checkpoint saved to models\checkpoints\policy_ep1500.pt
```

### 4 · Copy checkpoint to Pi and fine-tune on hardware

```powershell
# Copy checkpoint from Windows to Pi
scp models\checkpoints\policy_ep1500.pt pi@<PI_IP>:~/robot_car_claude/Agent_car_Claude/models/checkpoints/
```

```bash
# On Pi – fine-tune with real sensors/motors (~20 episodes)
python main.py --mode train --episodes 20 --hardware \
               --checkpoint models/checkpoints/policy_ep1500.pt
```

Place the robot in a clear ~2 m × 2 m area. The emergency stop fires
automatically if the ultrasonic reads < 10 cm.

### 5 · Run autonomously

```bash
python main.py --mode run --checkpoint models/checkpoints/policy_ep1500.pt
```

---

## RL State & Action Space

### State vector (439 float32 values with default config)

The HC-SR04 is a **fixed** forward-facing sensor — it does not sweep.
Only the camera (on the gimbal) can pan.

| Slice | Size | Description |
|---|---|---|
| Ultrasonic | 1 | Forward distance, normalised 0–1 |
| Visual features | 128 | MobileNetV2 bottleneck projection |
| Depth map | 256 | Flattened 16×16 monocular depth estimate |
| Local map | 49 | 7×7 occupancy probability window around robot |
| Heading | 2 | sin(θ), cos(θ) |
| Velocity | 3 | vx, vy, ω (body frame) |

### Action space (Discrete 11)

| Index | Action |
|---|---|
| 0 | Forward |
| 1 | Backward |
| 2 | Strafe left |
| 3 | Strafe right |
| 4 | Rotate left |
| 5 | Rotate right |
| 6 | Pan gimbal left |
| 7 | Pan gimbal right |
| 8 | Tilt gimbal up |
| 9 | Tilt gimbal down |
| 10 | Stop |

### Reward shaping

| Component | Value |
|---|---|
| New cell explored | +1.0 per cell |
| Collision | −15.0 |
| Close obstacle (<30 cm) | −0.5 / distance |
| Time step cost | −0.01 |
| Smooth motion bonus | +0.05 |
| Same action continuation | +0.025 |
| Reversal penalty | −0.05 |

---

## Training Results

Baseline sim training run (RTX 3050, cu126, ~24 min):

| Metric | Value |
|---|---|
| Episodes | 1500 |
| Final mean100 reward | 11 464 |
| Mean explored cells/ep | 13 648  (~34 m² at 0.05 m/cell) |
| Throughput | ~1 ep/s |

---

## Configuration Reference

All parameters are in `config/robot_config.yaml`.  No constants are
hardcoded in Python source — tune everything from the config file.

Key sections:
- `hal.*` — GPIO pins, I2C addresses, camera resolution, servo ranges
- `perception.*` — model weights, thresholds, depth output size
- `mapping.grid.*` — resolution, probabilistic update parameters
- `navigation.*` — A* heuristic, safety margin, pure-pursuit lookahead
- `rl.*` — PPO hyperparameters, state/action config, reward weights
- `agent.*` — loop frequency, frontier exploration settings

---

## Extending the System

**Add encoder odometry**
Implement `_update_pose()` in `rl/environment.py` using encoder
tick counts and call `_grid.update()` with the refined pose.

**Add a second ultrasonic sensor**
Extend `hal/ultrasonic.py` with a multi-sensor variant and update
`ray_angles` in `config/robot_config.yaml`.

**Swap the RL algorithm**
Replace `rl/agent.py` with a SAC or DQN implementation.
The `RobotEnv.step()` / `RobotEnv.reset()` interface is unchanged.

**Enable remote map monitoring**
Set `hal.camera.stream_enabled: true` and stream the annotated frame
from the debug overlay in `perception/vision.py`.
