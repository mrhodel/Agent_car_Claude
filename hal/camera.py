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

        self._cap     = None   # OpenCV VideoCapture (USB)
        self._picam   = None   # picamera2 instance
        self._lock    = threading.Lock()   # guards _latest_frame only
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._init_driver()

    # ── Initialisation ────────────────────────────────────────────

    def _init_driver(self) -> None:
        # Primary: USB camera (stock Raspbot V2 hardware)
        if self._driver == "usb":
            import cv2
            # On RPi 5 the internal camera interfaces claim low device indices,
            # so the USB camera may appear at video1, video2, etc.
            # Scan from the configured device_index up to device_index+10.
            _scan_range = 10
            cap = None
            found_index = None
            for idx in range(self._device, self._device + _scan_range):
                c = cv2.VideoCapture(idx)
                if c.isOpened():
                    # Configure BEFORE the first read so V4L2 doesn't need to
                    # reconfigure mid-stream (which causes frozen frames).
                    c.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
                    c.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                    c.set(cv2.CAP_PROP_FPS, self._fps)
                    # Verify it can actually produce a frame (rules out dummy devices)
                    ok, _ = c.read()
                    if ok:
                        cap = c
                        found_index = idx
                        break
                c.release()
            if cap is None or not cap.isOpened():
                logger.warning("[Camera] No USB camera found scanning /dev/video%d–%d -> sim",
                               self._device, self._device + _scan_range - 1)
                self._driver = "simulation"
                if cap:
                    cap.release()
            else:
                self._cap = cap
                self._device = found_index  # update so logs are accurate
                logger.info("[Camera] USB /dev/video%d  %dx%d @ %d fps",
                            found_index, self._width, self._height, self._fps)

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
        After 30 consecutive failures (~1 s) releases and reopens the device."""
        interval  = 1.0 / max(self._fps, 1)
        fail_count = 0

        while self._running:
            t0    = time.monotonic()
            frame = self._grab_frame()

            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
                fail_count = 0
            else:
                fail_count += 1
                if fail_count >= 30:
                    logger.warning("[Camera] V4L2 consecutive failures - reopening")
                    self._reopen()
                    fail_count = 0

            elapsed   = time.monotonic() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _reopen(self) -> None:
        """Release and reopen the USB camera after a stall."""
        import cv2
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        time.sleep(2.0)  # give V4L2 time to release the device
        # Scan all indices in case device node changed after reconnect
        for idx in range(self._device, self._device + 10):
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                c.set(cv2.CAP_PROP_FPS, self._fps)
                ok, _ = c.read()
                if ok:
                    self._cap = c
                    self._device = idx
                    logger.info("[Camera] Reopened /dev/video%d", idx)
                    return
            c.release()
        logger.warning("[Camera] Reopen failed - no working device found")

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
            if self._cap is None:
                return None   # stall recovery in progress; keep last good frame
            # Called only from _capture_loop (single thread) so no lock needed.
            ret, frame = self._cap.read()
            if not ret:
                logger.debug("[Camera] cap.read() returned False")
            return frame if ret else None

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
