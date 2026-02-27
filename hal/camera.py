"""
hal/camera.py
Camera driver for Yahboom Raspbot V2.

The Raspbot V2 is fitted with a 1 MP USB camera mounted on the 2-DOF
pan-tilt gimbal.  The primary driver is therefore ``usb`` (OpenCV
VideoCapture via V4L2).

Supported back-ends
-------------------
  • usb         – USB camera via OpenCV VideoCapture (default / primary)
  • picamera2   – Raspberry Pi CSI camera via libcamera (optional fallback
                  for use with an alternative CSI module, NOT the stock USB cam)
  • simulation  – generates synthetic noise frames for offline development

Frames are returned as BGR numpy arrays (OpenCV convention).
An optional MJPEG HTTP stream is available for remote monitoring.
"""
from __future__ import annotations

import logging
import threading
import time
import os
import zlib
import glob
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """
    Unified camera interface.

    Parameters
    ----------
    cfg : dict
        The ``hal.camera`` sub-dict from robot_config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        # Default driver is 'usb' – the stock Raspbot V2 camera is USB
        self._driver = cfg.get("driver", "usb")
        self._width   = int(cfg.get("width",  640))
        self._height  = int(cfg.get("height", 480))
        self._fps     = int(cfg.get("fps",    30))
        self._device  = int(cfg.get("device_index", 0))

        # Reopen heuristics (helps recover from UVC stalls/disconnects)
        self._freeze_reopen_s = float(cfg.get("freeze_reopen_s", 8.0))
        self._freeze_reopen_n = int(max(1, self._freeze_reopen_s * max(self._fps, 1)))

        self._usb_device_path: Optional[str] = None
        self._usb_failures = 0
        self._usb_last_reopen_t = 0.0

        # Protects access to self._cap during hot-swap/reopen
        self._cap_lock = threading.Lock()
        self._reopen_in_progress = False
        self._reopen_thread: Optional[threading.Thread] = None

        # Old capture handle pending release. Released by the capture thread to
        # avoid races with in-flight cap.read()/grab()/retrieve calls.
        self._cap_to_release = None

        self._cap     = None   # OpenCV VideoCapture (USB)
        self._picam   = None   # picamera2 instance
        self._lock    = threading.Lock()   # guards _latest_frame only
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_t = 0.0
        self._stale_last_warn_t = 0.0
        self._last_fp: int | None = None
        self._same_fp_count = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._init_driver()

    # ── Initialisation ────────────────────────────────────────────

    def _init_driver(self) -> None:
        # Primary: USB camera (stock Raspbot V2 hardware)
        if self._driver == "usb":
            import cv2

            # Prefer opening by *device path* (e.g. /dev/video0) instead of by
            # numeric index. On some OpenCV builds (notably on Raspberry Pi),
            # index-based open can produce misleading errors or fail even when
            # the V4L2 node is present.
            cfg_device_path = self._cfg.get("device_path")
            device_path = str(cfg_device_path) if cfg_device_path else None
            if device_path and not os.path.exists(device_path):
                logger.warning("[Camera] device_path not found: %s", device_path)
                device_path = None

            # Build a robust candidate list. USB cameras can renumber (/dev/videoN)
            # after disconnects; prefer stable /dev/v4l/by-id symlinks when available.
            candidates: list[str] = []
            if device_path:
                candidates.append(str(device_path))
                # If the user pinned a device_path, avoid scanning all /dev/video* nodes
                # (can be slow/hang on some nodes). Fall back to stable by-id/by-path only.
                stable = sorted(glob.glob('/dev/v4l/by-id/*video-index0'))
                if not stable:
                    # Only include USB/PCI by-path nodes; platform-* nodes are often ISP/pipeline devices and can block reads.
                    stable = [p for p in sorted(glob.glob('/dev/v4l/by-path/*video-index0')) if 'usb-' in p or 'pci-' in p]
                for p in stable:
                    if p not in candidates:
                        candidates.append(p)
            else:
                stable = sorted(glob.glob('/dev/v4l/by-id/*video-index0'))
                if not stable:
                    # Only include USB/PCI by-path nodes; platform-* nodes are often ISP/pipeline devices and can block reads.
                    stable = [p for p in sorted(glob.glob('/dev/v4l/by-path/*video-index0')) if 'usb-' in p or 'pci-' in p]
                for p in stable:
                    if p not in candidates:
                        candidates.append(p)
                def _is_usb_video_node(dev: str) -> bool:
                    real = os.path.realpath(dev)
                    if not real.startswith('/dev/video'):
                        return False
                    try:
                        idx = int(real.split('/dev/video', 1)[1])
                    except Exception:
                        return False
                    sys_dev = f'/sys/class/video4linux/video{idx}/device'
                    try:
                        return os.path.exists(sys_dev) and '/usb' in os.path.realpath(sys_dev)
                    except Exception:
                        return False

                preferred = f'/dev/video{self._device}'
                if os.path.exists(preferred) and preferred not in candidates:
                    candidates.append(preferred)
                # Finally, try every /dev/video* node that appears to be USB-backed.
                for dev in sorted(glob.glob('/dev/video*'), key=lambda s: (len(s), s)):
                    if dev in candidates:
                        continue
                    if _is_usb_video_node(dev):
                        candidates.append(dev)
            cap = None
            chosen = None

            for dev in candidates:
                c = self._open_usb_capture(dev)
                if c is not None:
                    cap = c
                    chosen = dev
                    break

            if cap is None or not cap.isOpened():
                if device_path:
                    logger.warning("[Camera] Failed to open USB camera at %s -> sim", device_path)
                else:
                    logger.warning(
                        "[Camera] No USB camera found scanning %s..%s -> sim",
                        candidates[0] if candidates else "/dev/video?",
                        candidates[-1] if candidates else "/dev/video?",
                    )
                self._driver = "simulation"
                if cap:
                    cap.release()
            else:
                self._cap = cap
                # Update device index if this resolves to /dev/videoN
                real = os.path.realpath(str(chosen))
                if real.startswith("/dev/video"):
                    try:
                        self._device = int(real.split("/dev/video", 1)[1])
                    except Exception:
                        pass
                self._usb_device_path = str(chosen)
                self._usb_failures = 0
                logger.info("[Camera] USB %s (%s)  %dx%d @ %d fps",
                            chosen, real, self._width, self._height, self._fps)


        # Optional: picamera2 for CSI-connected cameras (non-default on Raspbot V2)
        elif self._driver == "picamera2":
            try:
                from picamera2 import Picamera2
                self._picam = Picamera2()
                config = self._picam.create_preview_configuration(
                    main={"size": (self._width, self._height),
                          "format": "RGB888"},
                    controls={"FrameRate": self._fps}
                )
                self._picam.configure(config)
                self._picam.start()
                logger.info("[Camera] picamera2  %dx%d @ %d fps",
                            self._width, self._height, self._fps)
            except Exception as exc:
                logger.warning("[Camera] picamera2 unavailable (%s) -> USB", exc)
                self._driver = "usb"
                # Re-try as USB
                self._init_driver()
                return

        if self._driver == "simulation":
            logger.info("[Camera] Simulation mode  %dx%d", self._width, self._height)

        # Background thread: sole owner of cap.read() — keeps V4L2 buffer
        # continuously drained so _latest_frame always holds the newest frame.
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop,
                                        daemon=True, name="camera-capture")
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """
        Return the most recent frame as a BGR numpy array (H, W, 3).
        Returns None if no frame is available yet.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()


    def read_with_timestamp(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (frame, t_capture) where t_capture is from time.monotonic()."""
        with self._lock:
            if self._latest_frame is None:
                frame = None
                t_cap = 0.0
            else:
                frame = self._latest_frame.copy()
                t_cap = float(self._latest_frame_t)

        # Watchdog: if the capture loop is blocked (e.g., USB disconnect), the
        # stale timestamp is visible here even if the capture loop can't request
        # a reopen. This keeps the system recovering rather than hanging.
        if self._driver == "usb" and t_cap:
            try:
                stale_s = time.monotonic() - t_cap
                stale_reopen_s = float(self._cfg.get("stale_reopen_s", 1.5))
                if stale_s > stale_reopen_s:
                    self._request_reopen_usb(reason="stale")
            except Exception:
                pass

        return frame, t_cap
    def read_fresh(
        self,
        *,
        after_t: float,
        timeout_s: float = 0.6,
        poll_s: float = 0.005,
    ) -> Optional[np.ndarray]:
        """Wait for a frame captured strictly after after_t.

        This prevents higher-level logic (e.g., gimbal pan-then-look) from
        accidentally evaluating a cached pre-move frame.
        """
        deadline = time.monotonic() + float(max(0.0, timeout_s))
        while time.monotonic() < deadline:
            frame, t_cap = self.read_with_timestamp()
            if frame is not None and t_cap > after_t:
                return frame
            time.sleep(poll_s)
        return None

    def read_blocking(self, timeout_s: float = 1.0) -> Optional[np.ndarray]:
        """Block until a frame is available or timeout expires."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.005)
        return None

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._width, self._height

    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        if self._picam:
            self._picam.stop()
            self._picam.close()
        logger.info("[Camera] Closed")

    # ── Background capture loop ───────────────────────────────────

    def _capture_loop(self) -> None:
        """Sole caller of cap.read(). Rate-limited to camera fps.
        After any failures, keeps _latest_frame at last good value so callers
        see a valid stale frame rather than None. cap.read() recovers naturally
        once servo PWM noise subsides - no reopen needed."""
        interval = 1.0 / max(self._fps, 1)

        while self._running:
            t0    = time.monotonic()
            frame = self._grab_frame()

            if frame is not None:
                now = time.monotonic()
                with self._lock:
                    self._latest_frame = frame
                    self._latest_frame_t = now
                # Detect "stuck" camera output: sometimes V4L2 keeps returning
                # the same buffer repeatedly under load/disconnect.
                if self._driver == "usb":
                    fp = self._fingerprint_frame(frame)
                    if self._last_fp is not None and fp == self._last_fp:
                        self._same_fp_count += 1
                    else:
                        self._same_fp_count = 0
                        self._last_fp = fp
                    # If content appears frozen for too long, try a reopen.
                    # Keep this threshold fairly high to avoid false positives when the scene is static.
                    if self._same_fp_count >= self._freeze_reopen_n and (now - self._usb_last_reopen_t) > 2.0:
                        logger.warning("[Camera] Frame content frozen (~%ds) -> reopen", int(self._same_fp_count / max(self._fps,1)))
                        self._same_fp_count = 0
                        self._request_reopen_usb(reason="frozen")
            else:
                # If capture stalls for too long, reopen the USB device.
                if self._driver == "usb" and self._latest_frame_t:
                    now = time.monotonic()
                    stale_s = now - self._latest_frame_t
                    if stale_s > 1.5 and (now - self._usb_last_reopen_t) > 2.0:
                        if (now - self._stale_last_warn_t) > 5.0:
                            logger.warning("[Camera] No new frames for %.1fs -> reopen", stale_s)
                            self._stale_last_warn_t = now
                        self._request_reopen_usb(reason="stale")

            elapsed   = time.monotonic() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)


    def _fingerprint_frame(self, frame: np.ndarray) -> int:
        """Cheap content fingerprint for freeze detection."""
        # Sample a sparse grid (green channel) to keep this fast.
        sample = frame[::20, ::20, 1]
        return zlib.crc32(sample.tobytes())

    def _reopen(self) -> None:
        pass  # kept for API compatibility but no longer called

    def _open_usb_capture(self, dev: str):
        import cv2

        dev = str(dev)
        real = os.path.realpath(dev)

        probe_timeout_s = float(self._cfg.get('probe_read_timeout_s', 0.6))

        def _timed_read(cap):
            # Some V4L2 nodes can block in cap.read() for ~10s. Run read in a
            # daemon thread so probe never hangs startup/reopen.
            result: dict = {}

            def _reader():
                try:
                    result['ret'], result['frame'] = cap.read()
                except Exception as exc:
                    result['exc'] = exc

            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            t.join(timeout=probe_timeout_s)
            if t.is_alive():
                return False, None, True
            if 'exc' in result:
                return False, None, False
            return bool(result.get('ret')), result.get('frame'), False

        def _try(open_fn):
            cap = open_fn()
            if cap is None:
                return None
            if not cap.isOpened():
                cap.release()
                return None

            # Best-effort: ask OpenCV for shorter timeouts if supported.
            for prop_name, value in (
                ('CAP_PROP_OPEN_TIMEOUT_MSEC', int(probe_timeout_s * 1000)),
                ('CAP_PROP_READ_TIMEOUT_MSEC', int(probe_timeout_s * 1000)),
            ):
                prop = getattr(cv2, prop_name, None)
                if prop is not None:
                    try:
                        cap.set(prop, value)
                    except Exception:
                        pass

            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            ok = False
            timed_out = False
            frame = None
            for _ in range(6):
                ok, frame, timed_out = _timed_read(cap)
                if timed_out:
                    break
                if ok and frame is not None:
                    break
                time.sleep(0.05)

            if timed_out or not ok or frame is None:
                cap.release()
                return None
            return cap

        # Prefer opening by numeric index when possible. Some OpenCV builds on Pi
        # claim V4L2 is available but cannot capture "by name" (path), producing
        # noisy warnings and failing even for valid /dev/v4l/by-id symlinks.
        idx = None
        if real.startswith('/dev/video'):
            try:
                idx = int(real.split('/dev/video', 1)[1])
            except Exception:
                idx = None

        attempts = []
        if idx is not None:
            attempts.extend([
                lambda: cv2.VideoCapture(idx, cv2.CAP_V4L2),
                lambda: cv2.VideoCapture(idx),
            ])

        # Fallback to path opens only for direct /dev/videoN nodes.
        if dev.startswith('/dev/video'):
            attempts.extend([
                lambda: cv2.VideoCapture(dev, cv2.CAP_V4L2),
                lambda: cv2.VideoCapture(dev),
            ])
        if real != dev and real.startswith('/dev/video'):
            attempts.extend([
                lambda: cv2.VideoCapture(real, cv2.CAP_V4L2),
                lambda: cv2.VideoCapture(real),
            ])

        for a in attempts:
            cap = _try(a)
            if cap is not None:
                return cap

        return None
    def _request_reopen_usb(self, reason: str = "") -> None:
        """Request a USB reopen without blocking the capture loop.

        OpenCV/V4L2 open/probe can block for long periods when the device
        is disconnecting/reconnecting. Doing that work in the capture thread
        can stall *all* Python threads (GIL contention) and makes streaming
        appear to hang.
        """
        if self._driver != "usb":
            return
        now = time.monotonic()
        # Cooldown to avoid thrash
        if (now - self._usb_last_reopen_t) < 2.0:
            return
        if self._reopen_in_progress:
            return
        self._reopen_in_progress = True

        def _worker():
            try:
                if reason:
                    logger.info("[Camera] Reopen requested (%s)", reason)
                self._reopen_usb()
            finally:
                self._reopen_in_progress = False

        t = threading.Thread(target=_worker, daemon=True, name="camera-reopen")
        self._reopen_thread = t
        t.start()


    def _reopen_usb(self) -> None:
        if self._driver != "usb":
            return
        self._usb_last_reopen_t = time.monotonic()

        # Snapshot current cap under lock
        with self._cap_lock:
            old_cap = self._cap

        # When the camera disconnects, /dev and /dev/v4l can disappear for a
        # short time. Retry for a while so we can catch the re-enumeration.
        max_wait_s = float(self._cfg.get("reopen_max_wait_s", 6.0))
        deadline = time.monotonic() + max(0.0, max_wait_s)
        sleep_s = 0.2
        last_candidates: list[str] = []

        while True:
            cfg_device_path = self._cfg.get("device_path")
            device_path = str(cfg_device_path) if cfg_device_path else (self._usb_device_path or None)
            if device_path and not os.path.exists(str(device_path)):
                device_path = None

            # Try pinned/last-known first, then stable by-id/by-path. Avoid scanning
            # all /dev/video* when device_path is set (some nodes can block).
            candidates: list[str] = []
            if device_path:
                candidates.append(str(device_path))

            stable = sorted(glob.glob('/dev/v4l/by-id/*video-index0'))
            if not stable:
                stable = [
                    p for p in sorted(glob.glob('/dev/v4l/by-path/*video-index0'))
                    if 'usb-' in p or 'pci-' in p
                ]
            for p in stable:
                if p not in candidates:
                    candidates.append(p)

            if not device_path:
                def _is_usb_video_node(dev: str) -> bool:
                    real = os.path.realpath(dev)
                    if not real.startswith('/dev/video'):
                        return False
                    try:
                        idx = int(real.split('/dev/video', 1)[1])
                    except Exception:
                        return False
                    sys_dev = f'/sys/class/video4linux/video{idx}/device'
                    try:
                        return os.path.exists(sys_dev) and '/usb' in os.path.realpath(sys_dev)
                    except Exception:
                        return False

                for dev in sorted(glob.glob('/dev/video*'), key=lambda s: (len(s), s)):
                    if dev in candidates:
                        continue
                    if _is_usb_video_node(dev):
                        candidates.append(dev)

            last_candidates = candidates

            # Open new capture WITHOUT holding cap_lock (can block)
            new_cap = None
            new_dev = None
            for dev in candidates:
                cap = self._open_usb_capture(dev)
                if cap is not None:
                    new_cap = cap
                    new_dev = dev
                    break

            if new_cap is not None and new_dev is not None:
                # Swap under lock; defer release of old cap to capture thread.
                with self._cap_lock:
                    self._cap = new_cap
                    self._cap_to_release = old_cap

                self._usb_device_path = new_dev
                self._usb_failures = 0
                real = os.path.realpath(str(new_dev))
                if real.startswith('/dev/video'):
                    try:
                        self._device = int(real.split('/dev/video', 1)[1])
                    except Exception:
                        pass
                logger.info("[Camera] Reopened USB %s (%s)", new_dev, real)
                return

            if time.monotonic() >= deadline:
                break

            time.sleep(sleep_s)
            sleep_s = min(1.2, sleep_s * 1.6)

        # Keep the existing capture handle if we could not open a new one.
        with self._cap_lock:
            self._cap = old_cap
        logger.warning(
            "[Camera] Failed to reopen USB camera (%s)",
            last_candidates[0] if last_candidates else "none",
        )
    def _grab_frame(self) -> Optional[np.ndarray]:
        if self._driver == "picamera2" and self._picam:
            try:
                import cv2
                rgb = self._picam.capture_array()          # RGB888
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as exc:
                logger.debug("[Camera] picamera2 grab error: %s", exc)
                return None

        elif self._driver == "usb":
            with self._cap_lock:
                cap = self._cap
            if cap is None:
                return None   # stall recovery in progress; keep last good frame
            # Called only from _capture_loop (single thread) so no lock needed.
            # Drain any queued frames so callers see the most recent image
            try:
                for _ in range(2):
                    cap.grab()
                ret, frame = cap.retrieve()
            except Exception:
                ret, frame = cap.read()
            if not ret or frame is None:
                self._usb_failures += 1
                logger.debug("[Camera] cap.read() failed (n=%d)", self._usb_failures)
                # If the camera stops delivering after initial success,
                # reopen the device rather than streaming a stale frame forever.
                if self._usb_failures >= 30 and (time.monotonic() - self._usb_last_reopen_t) > 2.0:
                    logger.warning("[Camera] USB read failing repeatedly -> reopen")
                    self._request_reopen_usb(reason="read_fail")
                return None
            self._usb_failures = 0
            return frame

        else:  # simulation
            # Synthetic gradient + noise to exercise the vision pipeline
            frame = np.random.randint(50, 200,
                                      (self._height, self._width, 3),
                                      dtype=np.uint8)
            # Simulate a "floor" gradient at the bottom
            for row in range(self._height // 2, self._height):
                frame[row, :] = np.clip(
                    frame[row, :].astype(int) + 30, 0, 255).astype(np.uint8)
            return frame
