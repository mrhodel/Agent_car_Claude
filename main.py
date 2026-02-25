#!/usr/bin/env python3
"""
main.py  –  Yahboom Raspbot V2 Autonomous Navigation Agent
============================================================

Usage
-----
  # Run autonomously on the robot (loads last checkpoint if found)
  python main.py --mode run

  # Train in simulation (no hardware needed)
  python main.py --mode train --episodes 500

  # Train on the real robot
  python main.py --mode train --episodes 200 --hardware

  # Evaluate policy (greedy, no training)
  python main.py --mode eval --checkpoint models/checkpoints/policy_final.pt

  # Hardware self-test
  python main.py --mode test

Options
-------
  --config     Path to robot_config.yaml  [default: config/robot_config.yaml]
  --mode       run | train | eval | test | motion_test
  --episodes   Number of training episodes
  --hardware   Use real hardware during training (default: sim for train)
  --checkpoint Path to a .pt | .npz weights file to load
  --log-level  DEBUG | INFO | WARNING   [default: INFO]
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, "agent.log")),
    ]
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers)


def run_mode(cfg: dict, args: argparse.Namespace) -> None:
    """Full autonomous navigation."""
    from agent import AgentOrchestrator
    orchestrator = AgentOrchestrator(cfg)
    orchestrator.run(checkpoint=args.checkpoint)


def train_mode(cfg: dict, args: argparse.Namespace) -> None:
    """PPO training – simulation or hardware."""
    from mapping      import OccupancyGrid
    from perception   import VisionPipeline, SensorFusion
    from rl           import RobotEnv, PPOAgent, Trainer
    from rl.policy    import ActorCritic

    mode = "robot" if args.hardware else "sim"
    logging.getLogger().info("Training mode: %s", mode)

    rl_cfg    = cfg.get("rl", {})
    state_cfg = rl_cfg.get("state", {})

    # Build state dimension
    n_rays     = int(state_cfg.get("ultrasonic_rays", 5))
    feat_dim   = int(state_cfg.get("visual_feature_dim", 128))
    depth_flat = int(state_cfg.get("depth_map_flat", 256))
    map_sz     = int(state_cfg.get("local_map_size", 7))
    extra = 0
    if state_cfg.get("include_velocity", True): extra += 3
    if state_cfg.get("include_heading",  True): extra += 2
    state_dim = n_rays + feat_dim + depth_flat + map_sz**2 + extra

    n_actions = 11

    grid = OccupancyGrid(cfg.get("mapping", {}).get("grid", {}))

    # Minimal hardware for sim mode
    motors = us = gimbal = camera = vision = controller = None
    if mode == "robot":
        from hal        import MotorController, UltrasonicSensor, Gimbal, Camera
        from navigation import MotionController
        hal_cfg = cfg.get("hal", {})
        motors     = MotorController(hal_cfg.get("motor",      {}))
        us         = UltrasonicSensor(hal_cfg.get("ultrasonic", {}))
        gimbal     = Gimbal(hal_cfg.get("gimbal",     {}))
        camera     = Camera(hal_cfg.get("camera",     {}))
        vision     = VisionPipeline(cfg.get("perception", {}))
        controller = MotionController(motors,
                                       cfg.get("navigation", {}).get("controller", {}))

    env = RobotEnv(
        cfg=cfg, motors=motors, ultrasonic=us,
        gimbal=gimbal, camera=camera, vision=vision,
        grid=grid, controller=controller, mode=mode,
    )

    policy = ActorCritic(
        state_dim=state_dim, n_actions=n_actions,
        hidden_dim=int(rl_cfg.get("ppo", {}).get("hidden_dim", 256)),
    )
    ppo_cfg = dict(rl_cfg.get("ppo", {}))
    if mode == "robot" and "hardware_rollout_steps" in ppo_cfg:
        ppo_cfg["rollout_steps"] = ppo_cfg["hardware_rollout_steps"]
        logging.getLogger().info("Hardware mode: rollout_steps -> %d",
                                  ppo_cfg["rollout_steps"])
    agent = PPOAgent(policy=policy, cfg=ppo_cfg,
                     state_dim=state_dim, n_actions=n_actions)

    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load_checkpoint(args.checkpoint)

    trainer = Trainer(cfg, env, agent)
    try:
        trainer.train(n_episodes=args.episodes)
        results = trainer.evaluate(n_episodes=10)
        logging.getLogger().info("Final eval: %s", results)
    except (KeyboardInterrupt, SystemExit):
        logging.getLogger().warning("[Train] Interrupted — stopping hardware")
    finally:
        if mode == "robot":
            if motors:
                motors.stop()
                motors.close()
            if camera: camera.close()
            if gimbal: gimbal.close()


def eval_mode(cfg: dict, args: argparse.Namespace) -> None:
    """Greedy evaluation of a saved policy."""
    train_mode_args = argparse.Namespace(
        hardware=False, episodes=0, checkpoint=args.checkpoint)
    # Reuse train setup but only evaluate
    from mapping   import OccupancyGrid
    from rl        import RobotEnv, PPOAgent, Trainer
    from rl.policy import ActorCritic

    rl_cfg    = cfg.get("rl", {})
    state_cfg = rl_cfg.get("state", {})
    n_rays    = int(state_cfg.get("ultrasonic_rays", 5))
    feat_dim  = int(state_cfg.get("visual_feature_dim", 128))
    depth_flat= int(state_cfg.get("depth_map_flat", 256))
    map_sz    = int(state_cfg.get("local_map_size", 7))
    extra = 0
    if state_cfg.get("include_velocity", True): extra += 3
    if state_cfg.get("include_heading",  True): extra += 2
    state_dim = n_rays + feat_dim + depth_flat + map_sz**2 + extra

    grid   = OccupancyGrid(cfg.get("mapping", {}).get("grid", {}))
    env    = RobotEnv(cfg=cfg, grid=grid, mode="sim")
    policy = ActorCritic(state_dim=state_dim, n_actions=env.n_actions,
                          hidden_dim=int(rl_cfg.get("ppo", {})
                                          .get("hidden_dim", 256)))
    agent  = PPOAgent(policy=policy, cfg=rl_cfg.get("ppo", {}),
                      state_dim=state_dim, n_actions=env.n_actions)

    if args.checkpoint:
        agent.load_checkpoint(args.checkpoint)

    trainer = Trainer(cfg, env, agent)
    results = trainer.evaluate(n_episodes=args.episodes or 20)
    print("\nEvaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")


def motion_test_mode(cfg: dict) -> None:
    """
    Interactive motion test: step through every movement direction one at a
    time, display the expected wheel pattern, run the move for a short burst,
    then prompt the user to score the result.  A summary is printed at the end.

    Mecanum wheel layout (top view, arrows show roller spin direction):
         FL ── FR
         |      |
         RL ── RR
    ↑ = forward spin, ↓ = reverse spin, · = stopped
    """
    from hal import MotorController
    import time

    hal_cfg = cfg.get("hal", {})
    speed   = 50   # % – visible but not aggressive
    dur     = 1.2  # seconds per burst

    # (label, method_name_or_lambda, expected wheel pattern string)
    MOVES = [
        ("Forward",       lambda m: m.move_forward(speed),
         "FL↑  FR↑\nRL↑  RR↑"),
        ("Backward",      lambda m: m.move_backward(speed),
         "FL↓  FR↓\nRL↓  RR↓"),
        ("Strafe Right",  lambda m: m.strafe_right(speed),
         "FL↑  FR↓\nRL↓  RR↑"),
        ("Strafe Left",   lambda m: m.strafe_left(speed),
         "FL↓  FR↑\nRL↑  RR↓"),
        ("Rotate Left",   lambda m: m.rotate_left(speed),
         "FL↓  FR↑\nRL↓  RR↑"),
        ("Rotate Right",  lambda m: m.rotate_right(speed),
         "FL↑  FR↓\nRL↑  RR↓"),
    ]

    motors = MotorController(hal_cfg.get("motor", {}))
    results = []

    print("\n=== Motion Test ===")
    print(f"Speed: {speed}%  Duration: {dur}s per move")
    print("Press Enter to start each move, then rate it.\n")

    try:
        for label, move_fn, wheel_pattern in MOVES:
            print(f"─── {label} ───")
            print(f"Expected wheel pattern:\n{wheel_pattern}")
            input("  Press Enter to run … ")

            move_fn(motors)
            time.sleep(dur)
            motors.stop()
            time.sleep(0.3)

            while True:
                rating = input("  Result  [p]ass / [f]ail / [s]kip / [r]epeat, optional note: ").strip()
                if not rating:
                    rating = "p"
                if rating[0].lower() == "r":
                    print("  Repeating …")
                    move_fn(motors)
                    time.sleep(dur)
                    motors.stop()
                    time.sleep(0.3)
                    continue
                break
            tag    = rating[0].lower()
            note   = rating[2:].strip() if len(rating) > 1 else ""
            status = {"p": "PASS", "f": "FAIL", "s": "SKIP"}.get(tag, rating)
            results.append((label, status, note))
            print()

    except KeyboardInterrupt:
        motors.stop()
        print("\nAborted.")

    motors.close()

    print("=== Results ===")
    passed = failed = skipped = 0
    for label, status, note in results:
        suffix = f"  ({note})" if note else ""
        print(f"  {status:<4}  {label}{suffix}")
        if status == "PASS":   passed  += 1
        elif status == "FAIL": failed  += 1
        else:                  skipped += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped out of {len(MOVES)} moves.")


def test_mode(cfg: dict) -> None:
    """Hardware self-test: move each component briefly."""
    from hal import MotorController, UltrasonicSensor, Gimbal, Camera
    import time

    hal_cfg = cfg.get("hal", {})
    print("=== Hardware Self-Test ===")

    # Motors
    print("[1/4] Motor test …")
    motors = MotorController(hal_cfg.get("motor", {}))
    motors.move_forward(40); time.sleep(0.5)
    motors.move_backward(40); time.sleep(0.5)
    motors.stop()
    motors.close()
    print("      OK")

    # Gimbal
    print("[2/4] Gimbal test …")
    gimbal = Gimbal(hal_cfg.get("gimbal", {}))
    gimbal.set_pan(-45); time.sleep(0.4)
    gimbal.set_pan( 45); time.sleep(0.4)
    gimbal.centre()
    gimbal.close()
    print("      OK")

    # Ultrasonic
    print("[3/4] Ultrasonic test … (Ctrl+C to stop)")
    us = UltrasonicSensor(hal_cfg.get("ultrasonic", {}))
    try:
        while True:
            dist = us.read_cm()
            print(f"\r      Distance: {dist:6.1f} cm    ", end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print()
    us.close()
    print("      OK")

    # Camera
    print("[4/4] Camera test …")
    cam = Camera(hal_cfg.get("camera", {}))
    time.sleep(1.0)
    frame = cam.read_blocking(timeout_s=2.0)
    cam.close()
    if frame is not None:
        print(f"      Frame: {frame.shape}")
    else:
        print("      No frame received")

    print("=== Self-test complete ===")


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Yahboom Raspbot V2 Autonomous Navigation Agent")
    parser.add_argument("--config",     default="config/robot_config.yaml")
    parser.add_argument("--mode",       default="run",
                        choices=["run", "train", "eval", "test", "motion_test"])
    parser.add_argument("--episodes",   type=int, default=500)
    parser.add_argument("--hardware",   action="store_true",
                        help="Use real hardware during training")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to load a saved policy checkpoint")
    parser.add_argument("--log-level",  default="INFO")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_dir   = cfg.get("agent", {}).get("logging", {}).get("log_dir", "logs/")
    log_level = (cfg.get("agent", {}).get("logging", {}).get("level", args.log_level)
                 or args.log_level)
    setup_logging(log_level, log_dir)

    # Convert SIGTERM (pkill) into KeyboardInterrupt so finally blocks always run
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    logging.getLogger().info(
        "Yahboom Raspbot V2 Agent  mode=%s  config=%s", args.mode, args.config)

    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("models",             exist_ok=True)

    if args.mode == "run":
        run_mode(cfg, args)
    elif args.mode == "train":
        train_mode(cfg, args)
    elif args.mode == "eval":
        eval_mode(cfg, args)
    elif args.mode == "test":
        test_mode(cfg)
    elif args.mode == "motion_test":
        motion_test_mode(cfg)


if __name__ == "__main__":
    main()
