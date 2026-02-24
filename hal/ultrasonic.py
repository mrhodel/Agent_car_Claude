"""
hal/ultrasonic.py
HC-SR04 ultrasonic distance sensor driver for Yahboom Raspbot V2.

Hardware notes
--------------
The HC-SR04 sensor is mounted at a FIXED position on the robot chassis.
It always points forward (0° relative to the robot heading) and does NOT
move with the gimbal.  The entire sensor interface is routed through the
Yahboom MCU expansion board – no GPIO pins on the Raspberry Pi are used
for the ultrasonic sensor.

A median filter over ``samples_per_reading`` acquisitions is applied to
suppress spurious spikes.
"""
from __future__ import annotations

import logging
import statistics
import time
from typing import List

from .yahboom_board import YahboomBoard

logger = logging.getLogger(__name__)


class UltrasonicSensor:
    """
    Single fixed HC-SR04 sensor accessed via the Yahboom expansion board.

    The sensor always reads the distance straight ahead of the robot.
    There is no gimbal sweep capability; the gimbal is for the camera only.

    Parameters
    ----------
    cfg   : dict  –  the ``hal.ultrasonic`` sub-dict from robot_config.yaml
    board : YahboomBoard  –  shared board instance (may be None; one is
                             created internally if not provided)
    """

    def __init__(self, cfg: dict, board: YahboomBoard = None) -> None:
        self._cfg       = cfg
        self._max_range = float(cfg.get("max_range_cm", 400))
        self._min_range = float(cfg.get("min_range_cm", 2))
        self._n_samples = int(cfg.get("samples_per_reading", 3))
        self._sim_distance = 200.0   # cm, default simulation value

        # Close-range blind-spot guard: sensor returns max_range (invalid)
        # for objects < ~7 cm.  Track the last valid close reading so a
        # sudden jump to 400 doesn't look like a clear path.
        self._last_close_cm: float = self._max_range
        self._last_close_time: float = 0.0
        self._close_hysteresis_cm: float = 25.0  # cm threshold to watch
        self._close_hysteresis_s:  float = 2.0   # seconds to hold reading

        if board is not None:
            self._board = board
            logger.info("[Ultrasonic] Shared board, max_range=%.0f cm", self._max_range)
        else:
            i2c_bus  = int(cfg.get("i2c_bus",  1))
            i2c_addr = int(str(cfg.get("i2c_address", "0x7A")), 16)
            self._board = YahboomBoard(i2c_bus=i2c_bus, i2c_addr=i2c_addr)
            logger.info("[Ultrasonic] Own board (bus=%d addr=0x%02X)",
                        i2c_bus, i2c_addr)

    # ── Public API ────────────────────────────────────────────────

    def read_cm(self) -> float:
        """
        Return a median-filtered distance in centimetres.

        The sensor is fixed on the chassis and always measures forward.
        Multiple samples are averaged to reduce noise.

        Minimum reliable range is ~7 cm; below that the sensor returns
        max_range (hardware limitation).  A hysteresis guard substitutes
        the last valid close reading when this happens.
        """
        samples: List[float] = []
        raw: List[float] = []
        for _ in range(self._n_samples):
            d = self._board.get_distance()
            raw.append(d)
            if self._min_range <= d < self._max_range:
                samples.append(d)
            time.sleep(0.01)   # 10 ms between acquisitions

        if not samples:
            # All readings were rejected.  Two distinct failure modes:
            # (a) d < min_range → sensor blind spot (returns 0 or other sub-min
            #     garbage); object IS there, very close.
            #     Return min_range as a proxy for "closer than measurable".
            # (b) d >= max_range → nothing detected, or hardware error.
            #     Fall back to hysteresis / max_range.
            if any(d < self._min_range for d in raw):
                close_proxy = self._min_range
                self._last_close_cm   = close_proxy
                self._last_close_time = time.time()
                logger.debug("[Ultrasonic] Blind-spot: returning min_range %.1f cm",
                             close_proxy)
                return close_proxy
            # Hysteresis: if we recently saw something close, it's still there.
            if (self._last_close_cm < self._close_hysteresis_cm and
                    time.time() - self._last_close_time < self._close_hysteresis_s):
                logger.debug("[Ultrasonic] Hysteresis: returning %.1f cm",
                             self._last_close_cm)
                return self._last_close_cm
            return self._max_range

        result = statistics.median(samples)
        if result < self._close_hysteresis_cm:
            self._last_close_cm  = result
            self._last_close_time = time.time()
        return result

    def set_sim_distance(self, cm: float) -> None:
        """
        Override the value returned by get_distance() in simulation mode.
        Useful for offline RL training.
        """
        self._sim_distance = cm
        self._board._sim_distance = cm   # propagate to board stub if present

    def close(self) -> None:
        logger.debug("[Ultrasonic] Closed")

    # ── (no _single_shot needed – board handles timing) ──────────

    # Placeholder to satisfy any old call-sites that might reference
    # this attribute; the board owns the simulation state.
    @property
    def simulation(self) -> bool:
        return self._board._sim

    # ── Internal ─────────────────────────────────────────────────

    def _single_shot(self) -> None:
        raise NotImplementedError(
            "Direct GPIO is not used; call read_cm() which delegates to "
            "the YahboomBoard I2C driver.  The HC-SR04 is wired to the "
            "Yahboom expansion board, not to RPi GPIO pins directly."
        )


