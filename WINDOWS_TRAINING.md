# Windows GPU Training Guide

Train the PPO navigation agent on a Windows machine with an NVIDIA GPU, then transfer the checkpoint back to the Raspberry Pi.

---

## Prerequisites

- Python 3.10+ installed on Windows and on PATH
- NVIDIA GPU with CUDA (tested: RTX 3050, CUDA 13.0)
- SSH access to the Pi (Pi IP: `10.0.0.183`, user: `pi`)
- OpenSSH installed on Windows (comes with Windows 10/11)

---

## Step 1 – Copy files from Pi to Windows

Run in PowerShell:

```powershell
scp -r pi@10.0.0.183:~/robot_car_claude/Agent_car_Claude C:\Users\mhode\
cd C:\Users\mhode\Agent_car_Claude
```

---

## Step 2 – Create virtual environment

```powershell
python -m venv robot_venv
```

If activation is blocked by execution policy, run this **once**:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate:

```powershell
.\robot_venv\Scripts\Activate.ps1
```

---

## Step 3 – Install dependencies

```powershell
pip install -r requirements_desktop.txt
```

Install PyTorch with CUDA support.  
Use `cu126` for CUDA 12.x / 13.x (RTX 3050 with driver supporting CUDA 13.0):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> **Note:** If you have a different CUDA version, find the right wheel at https://pytorch.org/get-started/locally/

Verify GPU is detected:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True  NVIDIA GeForce RTX 3050`

---

## Step 4 – Run training

```powershell
python main.py --mode train --episodes 1500
```

Expected throughput: ~1 episode/second → ~25 minutes for 1500 episodes.

Log output example:
```
INFO  rl.agent   [PPO] ep=80  R=10636.6  mean100=9869.9  len=500
INFO  rl.trainer [Trainer] ep=80/1500  step=40000  72 s elapsed
```

Checkpoints are saved every 50 episodes to `models\checkpoints\policy_ep<N>.pt`.

---

## Step 5 – Copy trained model back to Pi

```powershell
scp models\checkpoints\policy_ep1500.pt pi@10.0.0.183:~/robot_car_claude/Agent_car_Claude/models/checkpoints/
```

---

## Step 6 – Fine-tune on real hardware (on Pi)

SSH into the Pi, then:

```bash
cd ~/robot_car_claude/Agent_car_Claude
python main.py --mode train --episodes 20 --hardware \
               --checkpoint models/checkpoints/policy_ep1500.pt
```

This runs the real motors/sensors for a short fine-tuning session to bridge the sim-to-real gap.

---

## Step 7 – Run autonomously (on Pi)

```bash
python main.py --mode run --checkpoint models/checkpoints/policy_ep1500.pt
```

---

## Training configuration

Key parameters in `config/robot_config.yaml` under `rl.ppo`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `episodes` | 1500 (CLI arg) | Total training episodes |
| `max_steps_per_episode` | 500 | Max env steps per episode |
| `rollout_steps` | 512 | PPO buffer size before update |
| `learning_rate` | 3e-4 | Adam LR |
| `hidden_dim` | 256 | Actor-critic MLP width |
| `checkpoint_interval` | 50 | Save every N episodes |

Under `mapping.grid`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `initial_size_cells` | 200 | 200×200 = 10m×10m map |
| `inflate_interval` | 10 | Gaussian blur every N steps (0=disabled) |

To disable obstacle inflation entirely for maximum speed (safe for sim-only training):

```yaml
mapping:
  grid:
    inflate_interval: 0
```

---

## Re-running training from a checkpoint

```powershell
python main.py --mode train --episodes 500 --checkpoint models\checkpoints\policy_ep1500.pt
```

---

## Troubleshooting

**`Activate.ps1` not recognised**  
→ Make sure you're in `C:\Users\mhode\Agent_car_Claude` and use `.\robot_venv\Scripts\Activate.ps1`

**`torch.cuda.is_available()` returns False**  
→ Wrong wheel installed. Uninstall and reinstall with the correct `cu126` URL above.

**UnicodeEncodeError in logs**  
→ All logger strings use ASCII only — should not occur. If it does, run on Pi:  
`grep -rn "logger\." . --include="*.py" | python3 -c "import sys; [print(l.rstrip()) for l in sys.stdin if not l.encode('cp1252', errors='ignore') == l.encode('cp1252')]"`
