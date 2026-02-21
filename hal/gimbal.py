"""
hal/gimbal.py
2-DOF pan-tilt gimbal controller for Yahboom Raspbot V2.

Hardware notes
--------------
The two servos (pan = horizontal, tilt = vertical) are driven by the
Yahboom MCU expansion board via I2C.  No direct RPi GPIO or PCA9685
chip is involved.

Servo IDs (Yahboom convention):
  1 = pan  (horizontal yaw)
  2 = tilt (vertical pitch)

Servo angles are in the Yahboom native range 0-180 degrees where 90° is
the physical centre.  Internally the class tracks angles relative to centre
(so 0° = straight ahead, negative = left/down, positive = right/up) and
converts to absolute 0-180 when calling the board.
"""
from __future__ import annotations

import logging
import time
from typing import Tuple

from .yahboom_board import YahboomBoard

logger = logging.getLogger(__name__)

# Yahboom servo ID assignment
_SERVO_PAN  = 1   # horizontal
_SERVO_TILT = 2   # vertical


class Gimbal:
    """
    Pan-tilt gimbal controller.

    Parameters
    ----------
    cfg   : dict  –  the ``hal.gimbal`` sub-dict from robot_config.yaml
    board : YahboomBoard (optional)  –  shared board; one is created if not given
    """

    def __init__(self, cfg: dict, board: YahboomBoard = None) -> None:
        self._cfg = cfg
        self._pan_range  = cfg.get("pan_range",  [-90, 90])
        self._tilt_range = cfg.get("tilt_range", [-45, 30])
        self._speed      = float(cfg.get("move_speed_deg_s", 120))

        # Current angles (degrees from centre)
        self._pan  = 0.0
        self._tilt = 0.0

        if board is not None:
            self._board = board
            logger.info("[Gimbal] Shared board")
        else:
            i2c_bus  = int(cfg.get("i2c_bus",  1))
            i2c_addr = int(str(cfg.get("i2c_address", "0x7A")), 16)
            self._board = YahboomBoard(i2c_bus=i2c_bus, i2c_addr=i2c_addr)
            logger.info("[Gimbal] Own board (bus=%d addr=0x%02X)",
                        i2c_bus, i2c_addr)

        # Move to centre on startup
        self.set_pan(0.0)
        self.set_tilt(0.0)

    # ── Public API ────────────────────────────────────────────────

    def set_pan(self, degrees: float) -> None:
        """Set pan angle (degrees from centre; 0 = forward)."""
        degrees = float(max(self._pan_range[0],
                            min(self._pan_range[1], degrees)))
        self._pan = degrees
        self._board.set_servo(_SERVO_PAN, self._to_absolute(degrees))

    def set_tilt(self, degrees: float) -> None:
        """Set tilt angle (degrees from horizontal; 0 = level)."""
        degrees = float(max(self._tilt_range[0],
                            min(self._tilt_range[1], degrees)))
        self._tilt = degrees
        self._board.set_servo(_SERVO_TILT, self._to_absolute(degrees))

    def move_pan(self, delta_deg: float) -> None:
        self.set_pan(self._pan + delta_deg)

    def move_tilt(self, delta_deg: float) -> None:
        self.set_tilt(self._tilt + delta_deg)

    def centre(self) -> None:
        self.set_pan(0.0)
        self.set_tilt(0.0)

    def sweep_pan(self, positions: list, dwell_s: float = 0.08) -> list:
        """
        Sweep the pan axis through a list of angles.
        Returns the list of positions actually visited (after clamping).
        Note: the camera is on the gimbal; the ultrasonic sensor is NOT.
        """
        visited = []
        for pos in positions:
            self.set_pan(pos)
            time.sleep(dwell_s)
            visited.append(self._pan)
        return visited

    @property
    def current_pan(self) -> float:
        return self._pan

    @property
    def current_tilt(self) -> float:
        return self._tilt

    @property
    def pose(self) -> Tuple[float, float]:
        return self._pan, self._tilt

    def close(self) -> None:
        self.centre()
        logger.debug("[Gimbal] Closed")

    # ── Internal ─────────────────────────────────────────────────

    @staticmethod
    def _to_absolute(degrees_from_centre: float) -> float:
        """
        Convert relative degrees (-90…+90 from centre) to Yahboom absolute
        servo angle (0-180, where 90 = centre).
        """
        return 90.0 + degrees_from_centre

