#!/usr/bin/env python3
"""Camera + chassis motors benchmark (no gimbal motion).

Purpose: test whether motor power draw / EMI correlates with USB camera stalls.
Motors are gated behind --enable-motors for safety.

Usage:
  python bench_camera_motors_only.py --duration 30
  python bench_camera_motors_only.py --duration 30 --enable-motors

Suggested kernel watch (separate terminal):
  sudo dmesg -w | egrep -i "usb|uvc|video|xHCI|dwc|over-current|under-voltage|thrott"
"""

from __future__ import annotations

import argparse
import time
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

from hal.camera import Camera
from hal.motors import MotorController
from hal.yahboom_board import YahboomBoard


@dataclass
class Stats:
    frames: int = 0
    none_frames: int = 0


def _fingerprint(frame_bgr: np.ndarray) -> int:
    small = frame_bgr[::16, ::16, :]
    return zlib.crc32(small.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/robot_config.yaml")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--print-every", type=float, default=1.0)
    ap.add_argument("--fps-target", type=float, default=30.0)
    ap.add_argument("--freeze-s", type=float, default=3.0)

    ap.add_argument("--enable-motors", action="store_true")
    ap.add_argument("--motor-speed", type=int, default=20)
    ap.add_argument("--segment", type=float, default=0.6)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    cam_cfg = cfg["hal"]["camera"]
    motor_cfg = cfg["hal"]["motor"]

    cam = Camera(cam_cfg)

    board: Optional[YahboomBoard] = None
    motors: Optional[MotorController] = None

    if args.enable_motors:
        board = YahboomBoard(i2c_bus=int(motor_cfg.get("i2c_bus", 1)), i2c_addr=int(str(motor_cfg.get("i2c_address", "0x7A")), 16))
        motors = MotorController(motor_cfg, board=board)
        motors.stop()
        print("[Motors] ENABLED", flush=True)
    else:
        print("[Motors] disabled (use --enable-motors)", flush=True)

    stats = Stats()
    last_fp: Optional[int] = None
    last_change_t = time.monotonic()

    speed = max(0, min(100, int(args.motor_speed)))
    segment = max(0.2, float(args.segment))
    motion = [
        ("fwd",   lambda: motors.move_forward(speed) if motors else None),
        ("stop", lambda: motors.stop() if motors else None),
        ("rotL", lambda: motors.rotate_left(max(10, speed)) if motors else None),
        ("stop", lambda: motors.stop() if motors else None),
        ("right",lambda: motors.strafe_right(speed) if motors else None),
        ("stop", lambda: motors.stop() if motors else None),
    ]

    t0 = time.monotonic()
    t_end = t0 + float(args.duration)
    next_print = t0 + float(args.print_every)

    sleep_s = 1.0 / max(1e-6, float(args.fps_target))

    i_motion = 0
    next_motion = t0
    motion_name = "idle"

    try:
        while True:
            now = time.monotonic()
            if now >= t_end:
                break

            if motors and now >= next_motion:
                motion_name, fn = motion[i_motion]
                fn()
                i_motion = (i_motion + 1) % len(motion)
                next_motion = now + segment

            frame = cam.read()
            if frame is None:
                stats.none_frames += 1
                time.sleep(min(0.01, sleep_s))
                continue

            stats.frames += 1
            fp = _fingerprint(frame)
            if last_fp is None:
                last_fp = fp
                last_change_t = now
            elif fp != last_fp:
                last_fp = fp
                last_change_t = now

            if now >= next_print:
                elapsed = now - t0
                fps = stats.frames / max(1e-6, elapsed)
                none_rate = stats.none_frames / max(1e-6, elapsed)
                frozen_for = now - last_change_t
                freeze_flag = " FREEZE?" if frozen_for >= float(args.freeze_s) else ""
                print(
                    f"t={elapsed:6.1f}s  fps={fps:5.1f}  none/s={none_rate:4.1f}  "
                    f"motion={motion_name:5s}  unchanged_for={frozen_for:4.1f}s{freeze_flag}",
                    flush=True,
                )
                next_print += float(args.print_every)

            time.sleep(sleep_s)

    finally:
        try:
            if motors:
                motors.stop()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass
        try:
            if board:
                board.close()
        except Exception:
            pass

    elapsed = time.monotonic() - t0
    fps = stats.frames / max(1e-6, elapsed)
    print(
        "\nSummary:\n"
        f"  duration_s={elapsed:.1f}\n"
        f"  frames={stats.frames}\n"
        f"  fps={fps:.2f}\n"
        f"  none_frames={stats.none_frames}\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
