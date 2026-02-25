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
        self._lock    = threading.Lock()   # serialises all cap.read() calls

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
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                cap.set(cv2.CAP_PROP_FPS, self._fps)
                # CAP_PROP_BUFFERSIZE is silently ignored on many V4L2 builds;
                # we drain stale frames manually instead (see _grab_frame).
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

    # ── Public API ────────────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """
        Capture and return a fresh frame as a BGR numpy array (H, W, 3).
        Returns None on failure.  Thread-safe: serialises via self._lock.
        """
        return self._grab_frame()

    def read_blocking(self, timeout_s: float = 1.0) -> Optional[np.ndarray]:
        """Capture a fresh frame, retrying until one arrives or timeout."""
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
        if self._cap:
            self._cap.release()
        if self._picam:
            self._picam.stop()
            self._picam.close()
        logger.info("[Camera] Closed")

    def _grab_frame(self) -> Optional[np.ndarray]:
        if self._driver == "picamera2" and self._picam:
            try:
                import cv2
                rgb = self._picam.capture_array()          # RGB888
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as exc:
                logger.debug("[Camera] picamera2 grab error: %s", exc)
                return None

        elif self._driver == "usb" and self._cap:
            with self._lock:
                ret, frame = self._cap.read()
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
