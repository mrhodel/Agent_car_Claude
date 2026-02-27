#!/usr/bin/env python3
"""Camera + gimbal sweep benchmark.

This adds I2C activity + servo motion while keeping the rest of the stack off.
Useful to test whether motion / EMI / power draw correlates with camera freezes.

Two key experiments:
  1) Small-range sweep (tests "any motion" without cable strain):
     python bench_camera_gimbal.py --duration 30 --pan-min -20 --pan-max 20 --pan-step 10

  2) End-stops only (tests cable strain / worst-case current spikes):
     python bench_camera_gimbal.py --duration 30 --endstop-only

Optional kernel watch (separate terminal):
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
from hal.gimbal import Gimbal
from hal.yahboom_board import YahboomBoard


@dataclass
class Stats:
    frames: int = 0
    none_frames: int = 0
    same_fp_frames: int = 0
    changed_fp_frames: int = 0


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

    ap.add_argument("--pan-step", type=float, default=30.0, help="Degrees between sweep positions")
    ap.add_argument("--dwell", type=float, default=0.12, help="Seconds between gimbal moves")
    ap.add_argument("--endstop-only", action="store_true", help="Move only between min/max (stress test)")

    ap.add_argument("--pan-min", type=float, default=None, help="Override pan minimum (deg from centre)")
    ap.add_argument("--pan-max", type=float, default=None, help="Override pan maximum (deg from centre)")
    ap.add_argument("--tilt", type=float, default=None, help="Optional fixed tilt angle (deg from centre)")

    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    cam_cfg = cfg["hal"]["camera"]
    gimbal_cfg = cfg["hal"]["gimbal"]

    board = YahboomBoard(
        i2c_bus=int(gimbal_cfg.get("i2c_bus", 1)),
        i2c_addr=int(str(gimbal_cfg.get("i2c_address", "0x7A")), 16),
    )
    cam = Camera(cam_cfg)
    gimbal = Gimbal(gimbal_cfg, board=board)

    if args.tilt is not None:
        gimbal.set_tilt(float(args.tilt))

    stats = Stats()
    last_fp: Optional[int] = None
    last_change_t = time.monotonic()

    cfg_pan_min, cfg_pan_max = gimbal_cfg.get("pan_range", [-90, 90])
    pan_min = float(cfg_pan_min if args.pan_min is None else args.pan_min)
    pan_max = float(cfg_pan_max if args.pan_max is None else args.pan_max)
    if pan_max < pan_min:
        pan_min, pan_max = pan_max, pan_min

    if args.endstop_only:
        positions = [pan_min, pan_max]
    else:
        step = max(1.0, float(args.pan_step))
        positions = list(np.arange(pan_min, pan_max + 0.1, step))
        if len(positions) <= 1:
            positions = [pan_min, pan_max]
        positions = positions + list(reversed(positions))

    print(
        f"[GimbalBench] pan_min={pan_min:.1f} pan_max={pan_max:.1f} positions={len(positions)} endstop_only={bool(args.endstop_only)}",
        flush=True,
    )

    t0 = time.monotonic()
    t_end = t0 + float(args.duration)
    next_print = t0 + float(args.print_every)

    sleep_s = 1.0 / max(1e-6, float(args.fps_target))

    i_pos = 0
    next_move = t0

    try:
        while True:
            now = time.monotonic()
            if now >= t_end:
                break

            if now >= next_move:
                gimbal.set_pan(float(positions[i_pos]))
                i_pos = (i_pos + 1) % len(positions)
                next_move = now + float(args.dwell)

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
            elif fp == last_fp:
                stats.same_fp_frames += 1
            else:
                stats.changed_fp_frames += 1
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
                    f"pan={gimbal.current_pan:6.1f}  unchanged_for={frozen_for:4.1f}s{freeze_flag}",
                    flush=True,
                )
                next_print += float(args.print_every)

            time.sleep(sleep_s)

    finally:
        try:
            gimbal.close()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass
        try:
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
        f"  none_frames={stats.none_frames}\n"
        f"  same_fp_frames={stats.same_fp_frames}\n"
        f"  changed_fp_frames={stats.changed_fp_frames}\n",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
