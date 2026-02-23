"""
hal/motors.py
Mecanum-wheel motor controller for Yahboom Raspbot V2.

All motor commands are forwarded to the Yahboom MCU expansion board via
the YahboomBoard I2C wrapper.  No direct GPIO or raw I2C usage here.

Motor layout (top view, front = ↑):
         FL ── FR
         |      |
         RL ── RR

Mecanum IK signs follow the standard +x = forward, +y = right, +ω = CCW
convention.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .yahboom_board import YahboomBoard

logger = logging.getLogger(__name__)


@dataclass
class MotorValues:
    fl: int  # front-left   duty-cycle  -100 … +100
    fr: int  # front-right
    rl: int  # rear-left
    rr: int  # rear-right

    def clamp(self, max_speed: int = 100) -> "MotorValues":
        def c(v: int) -> int:
            return max(-max_speed, min(max_speed, int(v)))
        return MotorValues(c(self.fl), c(self.fr), c(self.rl), c(self.rr))


class MotorController:
    """
    Thread-safe motor controller for Mecanum-wheeled Raspbot V2.

    Delegates all hardware communication to a shared YahboomBoard instance.

    Parameters
    ----------
    cfg   : dict  –  the ``hal.motor`` sub-dict from robot_config.yaml
    board : YahboomBoard (optional)  –  shared board; one is created if not given
    """

    def __init__(self, cfg: dict, board: YahboomBoard = None) -> None:
        self._cfg       = cfg
        self._max_speed = int(cfg.get("max_speed", 80))
        self._current   = MotorValues(0, 0, 0, 0)

        if board is not None:
            self._board = board
            logger.info("[Motors] Shared board, max_speed=%d", self._max_speed)
        else:
            i2c_bus  = int(cfg.get("i2c_bus",  1))
            i2c_addr = int(str(cfg.get("i2c_address", "0x7A")), 16)
            self._board = YahboomBoard(i2c_bus=i2c_bus, i2c_addr=i2c_addr)
            logger.info("[Motors] Own board (bus=%d addr=0x%02X)",
                        i2c_bus, i2c_addr)

    # ── Public API ────────────────────────────────────────────────

    def set_velocity(self, vx: float, vy: float, omega: float) -> None:
        """
        Command the chassis using a body-frame velocity vector.

        Parameters
        ----------
        vx    : forward  velocity  (-1 … +1, normalised)
        vy    : sideways velocity  (-1 … +1, right positive)
        omega : yaw rate           (-1 … +1, CCW positive)
        """
        mv = self._mecanum_ik(vx, vy, omega)
        mv = mv.clamp(self._max_speed)
        self._apply(mv)

    def move_forward(self,  speed: int = 50) -> None: self.set_velocity( speed / 100, 0, 0)
    def move_backward(self, speed: int = 50) -> None: self.set_velocity(-speed / 100, 0, 0)
    def strafe_left(self,   speed: int = 50) -> None: self.set_velocity(0, -speed / 100, 0)
    def strafe_right(self,  speed: int = 50) -> None: self.set_velocity(0,  speed / 100, 0)
    def rotate_left(self,   speed: int = 45) -> None: self.set_velocity(0, 0,  speed / 100)
    def rotate_right(self,  speed: int = 45) -> None: self.set_velocity(0, 0, -speed / 100)

    def stop(self) -> None:
        self._apply(MotorValues(0, 0, 0, 0))

    @property
    def current_values(self) -> MotorValues:
        return self._current

    def close(self) -> None:
        self.stop()
        logger.info("[Motors] Closed")

    # ── Mecanum inverse kinematics ────────────────────────────────

    @staticmethod
    def _mecanum_ik(vx: float, vy: float, omega: float) -> MotorValues:
        """
        Mecanum wheel IK matched to Yahboom Raspbot V2 physical wiring.

        Verified against official McLumk_Wheel_Sports.py:
          ID0 FL : +vx +vy -omega
          ID1 RL : +vx -vy -omega
          ID2 FR : +vx -vy +omega
          ID3 RR : +vx +vy +omega

        Convention: +vx = forward, +vy = strafe right, +omega = rotate left (CCW)
        """
        fl = ( vx + vy - omega) * 100.0
        fr = ( vx - vy + omega) * 100.0
        rl = ( vx - vy - omega) * 100.0
        rr = ( vx + vy + omega) * 100.0
        return MotorValues(int(fl), int(fr), int(rl), int(rr))

    # ── Board dispatch ────────────────────────────────────────────

    def _apply(self, mv: MotorValues) -> None:
        self._current = mv
        self._board.set_motors(mv.fl, mv.fr, mv.rl, mv.rr)
        logger.debug("[Motors] FL=%d FR=%d RL=%d RR=%d",
                     mv.fl, mv.fr, mv.rl, mv.rr)
