#!/usr/bin/env python3
"""
bench_us_test.py  –  Ultrasonic sensor characterisation tool
=============================================================

Run this with the robot on the bench, wheels off the ground.  The script
cycles through four test phases while continuously reading the HC-SR04 and
logs every reading to CSV and the terminal.

Phases
------
  1. IDLE         – no motion
  2. GIMBAL_SWEEP – pan left/right and tilt up/down continuously
  3. MOTORS_SPIN  – all four wheels spin at low speed (no load)
  4. COMBINED     – motors + gimbal moving at the same time

What to look for
----------------
  * Readings ≤ min_range_cm (3.0) while the robot is clearly far from any
    wall → genuine EMI / interference glitch.
  * Readings that jump by > JUMP_THRESHOLD_CM between consecutive samples
    → transient spikes.
  * Whether the glitch rate is significantly higher in phases 3 / 4 than
    phase 1 → motor or servo noise coupling into the sensor.

Usage
-----
  python bench_us_test.py                        # uses default config
  python bench_us_test.py --config config/robot_config.yaml
  python bench_us_test.py --duration 15          # seconds per phase
  python bench_us_test.py --motor-speed 50       # duty cycle 0-100
  python bench_us_test.py --no-gimbal            # skip gimbal phases
  python bench_us_test.py --no-motors            # skip motor phases

Output
------
  bench_us_test_YYYYMMDD_HHMMSS.csv   – every individual reading
  bench_us_test_YYYYMMDD_HHMMSS.log   – phase summaries
"""
from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import statistics
import sys
import time
from typing import List, Tuple

import yaml


# ── Colour helpers ────────────────────────────────────────────────────────────
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_RESET  = "\033[0m"


def _col(text: str, colour: str) -> str:
    """Apply ANSI colour only when stdout is a tty."""
    if sys.stdout.isatty():
        return f"{colour}{text}{_RESET}"
    return text


# ── Data collection ───────────────────────────────────────────────────────────
JUMP_THRESHOLD_CM = 20.0   # flag large inter-sample jumps


def collect_readings(us, phase_name: str, duration_s: float,
                     min_range_cm: float) -> List[Tuple[float, float, bool, bool]]:
    """
    Read the US sensor as fast as it will go for *duration_s* seconds.

    Returns list of (timestamp, distance_cm, is_min_range, is_jump).
    """
    rows: List[Tuple[float, float, bool, bool]] = []
    prev_cm: float | None = None
    t_end = time.monotonic() + duration_s
    glitch_count = 0
    jump_count = 0

    print(f"\n{_col(f'  Phase: {phase_name}', _CYAN)}  ({duration_s:.0f} s)")
    print(f"  {'Time':>8}  {'Dist cm':>9}  {'Flag'}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*14}")

    while time.monotonic() < t_end:
        ts = time.time()
        try:
            cm = us.read_cm()
        except Exception as exc:
            print(f"  {_col(f'  READ ERROR: {exc}', _RED)}")
            time.sleep(0.1)
            continue

        is_min = (cm <= min_range_cm)
        is_jump = (prev_cm is not None and abs(cm - prev_cm) > JUMP_THRESHOLD_CM)
        rows.append((ts, cm, is_min, is_jump))

        flag = ""
        colour = _GREEN
        if is_min:
            flag = "*** MIN_RANGE"
            colour = _RED
            glitch_count += 1
        elif is_jump:
            flag = f"~ JUMP from {prev_cm:.1f}"
            colour = _YELLOW
            jump_count += 1

        rel_t = ts - rows[0][0]
        line = f"  {rel_t:8.2f}  {cm:9.1f}  {flag}"
        print(_col(line, colour) if flag else line)
        prev_cm = cm

    # Phase summary
    dists = [r[1] for r in rows]
    if dists:
        summary = (
            f"\n  Summary [{phase_name}]: "
            f"n={len(dists)}  "
            f"min={min(dists):.1f}  max={max(dists):.1f}  "
            f"mean={statistics.mean(dists):.1f}  "
            f"stdev={statistics.stdev(dists) if len(dists) > 1 else 0:.2f}  "
            f"min_range_hits={glitch_count}  "
            f"jumps={jump_count}"
        )
        colour = _RED if glitch_count else (_YELLOW if jump_count else _GREEN)
        print(_col(summary, colour))
    return rows


# ── Gimbal sweep thread ───────────────────────────────────────────────────────
import threading


class _GimbalSweeper(threading.Thread):
    """Continuously pan left↔right and tilt up↔down until stopped."""
    def __init__(self, gimbal, pan_deg: float = 60, tilt_deg: float = 20):
        super().__init__(daemon=True)
        self.gimbal = gimbal
        self.pan_deg = pan_deg
        self.tilt_deg = tilt_deg
        self._stop = threading.Event()

    def run(self):
        g = self.gimbal
        while not self._stop.is_set():
            g.set_pan(-self.pan_deg)
            time.sleep(0.6)
            g.set_pan(self.pan_deg)
            time.sleep(0.6)
        g.centre()

    def stop(self):
        self._stop.set()
        self.join(timeout=3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="US sensor bench characterisation")
    ap.add_argument("--config",       default="config/robot_config.yaml")
    ap.add_argument("--duration",     type=float, default=10.0,
                    help="Seconds per phase (default: 10)")
    ap.add_argument("--motor-speed",  type=int,   default=40,
                    help="Motor duty cycle for spin phase 0-100 (default: 40)")
    ap.add_argument("--no-gimbal",    action="store_true")
    ap.add_argument("--no-motors",    action="store_true")
    ap.add_argument("--out-dir",      default="logs")
    args = ap.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    hal_cfg     = cfg.get("hal", {})
    us_cfg      = hal_cfg.get("ultrasonic", {})
    motor_cfg   = hal_cfg.get("motor", {})
    gimbal_cfg  = hal_cfg.get("gimbal", {})
    min_range   = float(us_cfg.get("min_range_cm", 3.0))

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, f"bench_us_test_{stamp}.log")
    csv_path = os.path.join(args.out_dir, f"bench_us_test_{stamp}.csv")
    logging.basicConfig(level=logging.WARNING)  # suppress HAL noise

    print(_col(f"\n=== US Bench Test  [{stamp}] ===", _CYAN))
    print(f"  Config      : {args.config}")
    print(f"  Duration    : {args.duration} s/phase")
    print(f"  Motor speed : {args.motor_speed}")
    print(f"  min_range_cm: {min_range}")
    print(f"  CSV out     : {csv_path}")
    print(f"  Log out     : {log_path}")
    print()
    print(_col("  ** Ensure wheels are OFF the ground before continuing **", _RED))
    input("  Press ENTER to start...")

    # ── Hardware init ─────────────────────────────────────────────────────────
    from hal.yahboom_board import YahboomBoard
    from hal.ultrasonic   import UltrasonicSensor
    from hal.motors       import MotorController
    from hal.gimbal       import Gimbal

    board  = YahboomBoard(int(us_cfg.get("i2c_bus", 1)),
                          int(str(us_cfg.get("i2c_address", "0x2B")), 16))
    us     = UltrasonicSensor(us_cfg, board=board)
    motors = MotorController(motor_cfg, board=board) if not args.no_motors else None
    gimbal = Gimbal(gimbal_cfg, board=board) if not args.no_gimbal else None

    if motors:
        motors.stop()
    if gimbal:
        gimbal.centre()
    time.sleep(0.5)

    # ── Test phases ───────────────────────────────────────────────────────────
    all_rows: List[dict] = []

    def _run_phase(name: str, setup=None, teardown=None):
        if setup:
            setup()
        rows = collect_readings(us, name, args.duration, min_range)
        if teardown:
            teardown()
        for (ts, cm, is_min, is_jump) in rows:
            all_rows.append({
                "phase":        name,
                "timestamp":    ts,
                "distance_cm":  cm,
                "is_min_range": int(is_min),
                "is_jump":      int(is_jump),
            })

    # Phase 1: Idle
    _run_phase("IDLE")

    # Phase 2: Gimbal sweep
    if gimbal and not args.no_gimbal:
        sweeper = _GimbalSweeper(gimbal)
        _run_phase("GIMBAL_SWEEP",
                   setup=sweeper.start,
                   teardown=sweeper.stop)
        gimbal.centre()
        time.sleep(0.3)

    # Phase 3: Motors spinning (no translation – all wheels at same speed
    #          so robot stays still if held level, or barely creeps)
    if motors and not args.no_motors:
        spd = args.motor_speed

        def _start_motors():
            motors.move_forward(spd)

        def _stop_motors():
            motors.stop()

        _run_phase("MOTORS_SPIN", setup=_start_motors, teardown=_stop_motors)
        time.sleep(0.3)

    # Phase 4: Combined
    if motors and gimbal and not args.no_gimbal and not args.no_motors:
        sweeper2 = _GimbalSweeper(gimbal)
        spd = args.motor_speed

        def _start_combined():
            motors.move_forward(spd)
            sweeper2.start()

        def _stop_combined():
            motors.stop()
            sweeper2.stop()
            gimbal.centre()

        _run_phase("COMBINED", setup=_start_combined, teardown=_stop_combined)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if motors:
        motors.stop()
    if gimbal:
        gimbal.centre()

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

    # ── Final report ──────────────────────────────────────────────────────────
    print(_col("\n=== FINAL REPORT ===", _CYAN))
    phases = sorted({r["phase"] for r in all_rows},
                    key=lambda p: ("IDLE","GIMBAL_SWEEP","MOTORS_SPIN","COMBINED").index(p)
                    if p in ("IDLE","GIMBAL_SWEEP","MOTORS_SPIN","COMBINED") else 99)
    with open(log_path, "w") as lf:
        for phase in phases:
            pr = [r for r in all_rows if r["phase"] == phase]
            dists     = [r["distance_cm"]  for r in pr]
            glitches  = sum(r["is_min_range"] for r in pr)
            jumps     = sum(r["is_jump"]      for r in pr)
            rate = len(dists) / args.duration
            line = (
                f"{phase:<16}  n={len(dists):4d}  rate={rate:5.1f}/s  "
                f"min={min(dists):6.1f}  max={max(dists):6.1f}  "
                f"mean={statistics.mean(dists):6.1f}  "
                f"stdev={statistics.stdev(dists) if len(dists)>1 else 0:.2f}  "
                f"min_range_hits={glitches:3d} ({100*glitches/len(dists):.1f}%)  "
                f"jumps={jumps:3d}"
            )
            colour = _RED if glitches else (_YELLOW if jumps else _GREEN)
            print(_col(f"  {line}", colour))
            lf.write(line + "\n")

    print(f"\n  CSV  → {csv_path}")
    print(f"  Log  → {log_path}\n")


if __name__ == "__main__":
    main()
