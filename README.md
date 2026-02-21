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
| Range sensor | HC-SR04 ultrasonic (1 transducer + gimbal sweep) |
| Camera | RPi Camera Module 3 or USB UVC camera |
| Gimbal | 2-DOF pan-tilt (PCA9685 servos) |
| Motor driver | Yahboom expansion board (I2C address 0x7A) |

---

## Project Structure

```
Agent_car_Claude/
├── config/
│   └── robot_config.yaml      # All tuneable parameters
├── hal/
│   ├── motors.py              # Mecanum wheel controller (I2C / GPIO)
│   ├── ultrasonic.py          # HC-SR04 with median filter + ray sweep
│   ├── gimbal.py              # PCA9685 pan-tilt controller
│   └── camera.py              # picamera2 / USB / sim back-end
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

### 1 · Install dependencies

```bash
# On Raspberry Pi 5
pip install -r requirements.txt

# Uncomment the RPi-specific lines in requirements.txt first:
#   RPi.GPIO, smbus2, picamera2, adafruit-circuitpython-pca9685
```

### 2 · Configure hardware

Edit `config/robot_config.yaml` to match your wiring:
```yaml
hal:
  motor:
    driver: yahboom_i2c       # or gpio_pwm / simulation
  ultrasonic:
    trig: 27
    echo: 22
  gimbal:
    driver: pca9685
    pan_channel: 0
    tilt_channel: 1
  camera:
    driver: picamera2         # or usb / simulation
```

### 3 · Hardware self-test

```bash
python main.py --mode test
```

### 4 · Train the RL policy (simulation, no hardware)

```bash
python main.py --mode train --episodes 500
# Checkpoints saved to models/checkpoints/
```

### 5 · Fine-tune on the real robot

```bash
python main.py --mode train --episodes 100 --hardware \
               --checkpoint models/checkpoints/policy_ep500.pt
```

### 6 · Run autonomously

```bash
python main.py --mode run \
               --checkpoint models/checkpoints/policy_final.pt
```

---

## RL State & Action Space

### State vector (443 float32 values with default config)

| Slice | Size | Description |
|---|---|---|
| Ultrasonic rays | 5 | Normalised distances at [-60°,-30°,0°,+30°,+60°] |
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
