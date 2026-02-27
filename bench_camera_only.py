#!/usr/bin/env python3
"""Standalone camera benchmark.

Goal: isolate camera/USB stability without MJPEG, RL, Torch, or any other load.

Typical usage:
  python bench_camera_only.py --duration 60

Tips:
  In another terminal, watch kernel USB/UVC events:
    sudo dmesg -w | egrep -i "usb|uvc|video|xHCI|dwc|over-current|under-voltage|thrott"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

from hal.camera import Camera


@dataclass
class Stats:
    frames: int = 0
    none_frames: int = 0
    same_fp_frames: int = 0
    changed_fp_frames: int = 0


def _fingerprint(frame_bgr: np.ndarray) -> int:
    # Robust + cheap: downsample heavily, then CRC32.
    small = frame_bgr[::16, ::16, :]
    return zlib.crc32(small.tobytes())


def _try_vcgencmd_get_throttled() -> Optional[str]:
    if shutil.which("vcgencmd") is None:
        return None
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
        return out
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/robot_config.yaml")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--print-every", type=float, default=1.0)
    ap.add_argument("--fps-target", type=float, default=30.0, help="Read loop pacing; not camera driver FPS")
    ap.add_argument("--freeze-s", type=float, default=3.0, help="Report if identical fingerprint persists this long")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    cam_cfg = cfg["hal"]["camera"]

    thrott0 = _try_vcgencmd_get_throttled()
    if thrott0 is not None:
        print(f"[Pi] vcgencmd get_throttled (start): {thrott0}")

    cam = Camera(cam_cfg)

    stats = Stats()
    last_fp: Optional[int] = None
    last_change_t = time.monotonic()

    t0 = time.monotonic()
    t_end = t0 + float(args.duration)
    next_print = t0 + float(args.print_every)

    sleep_s = 1.0 / max(1e-6, float(args.fps_target))

    try:
        while True:
            now = time.monotonic()
            if now >= t_end:
                break

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
                    f"same={stats.same_fp_frames:6d}  changed={stats.changed_fp_frames:6d}  "
                    f"unchanged_for={frozen_for:4.1f}s{freeze_flag}",
                    flush=True,
                )
                next_print += float(args.print_every)

            time.sleep(sleep_s)

    finally:
        cam.close()

    thrott1 = _try_vcgencmd_get_throttled()
    if thrott1 is not None:
        print(f"[Pi] vcgencmd get_throttled (end):   {thrott1}")

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
