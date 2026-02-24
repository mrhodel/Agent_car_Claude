"""
hal/yahboom_board.py
Thin compatibility wrapper around the Yahboom expansion board driver.

Resolution order
----------------
1. Try to import the pre-installed ``Raspbot`` Python package that ships
   on the Yahboom SD-card image (typical install path: /home/pi/Raspbot.py
   or installed system-wide via the Yahboom setup script).
2. Fall back to a direct ``smbus2`` I2C implementation that speaks the
   same wire protocol as the Yahboom MCU expansion board.
3. If neither is available (no I2C bus / desktop development), fall back
   to a pure-simulation stub that logs every call and returns safe values.

Usage
-----
    from hal.yahboom_board import YahboomBoard
    board = YahboomBoard(i2c_bus=1, i2c_addr=0x7A)
    board.set_motors(60, 60, 60, 60)   # FL, FR, RL, RR  (-100…+100)
    board.set_servo(1, 90)              # servo_id (1-based), angle (0-180)
    dist = board.get_distance()         # cm  (float)
    board.stop_motors()

Wire-protocol (I2C, addr 0x7A)
-------------------------------
The Yahboom MCU accepts multi-byte writes and replies:

  Motor command:
    [0x01, fl+100, fr+100, rl+100, rr+100]   # encode -100…100 → 0…200
    Each byte 0x00 = full reverse, 0x64 = stop, 0xC8 = full forward

  Servo command:
    [0x10, servo_id, angle_hi, angle_lo]      # angle 0-1000 (10× degrees)
    servo_id:  1 = pan (horizontal), 2 = tilt (vertical)
    angle:     0-1000 maps to 0°-100°  (multiply degrees by 10)
               Yahboom uses 0-1800 for 0°-180°

  Ultrasonic read:
    Write [0x02] then read 3 bytes → [hi, mid, lo] → distance = hi<<16|mid<<8|lo  mm
    (convert mm → cm by dividing by 10)

  Stop all:
    [0x01, 100, 100, 100, 100]  (all channels at stop)

Note: These constants are derived from reverse-engineering the publicly
available Yahboom PDF courses and are approximate.  If the pre-installed
Raspbot library is found, that takes precedence over this raw protocol.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── I2C register map ──────────────────────────────────────────────
_REG_MOTOR     = 0x01   # motor speeds
_REG_ULTRASONIC= 0x02   # ultrasonic trigger/read
_REG_SERVO     = 0x10   # servo position
_MOTOR_STOP    = 100    # encoded value for 0% duty (range 0-200)

# How many ms to wait after the ultrasonic trigger before reading
_US_TRIGGER_WAIT_S = 0.025


class YahboomBoard:
    """
    Unified interface to the Yahboom expansion board.

    Parameters
    ----------
    i2c_bus  : Raspberry Pi I2C bus number (default 1 for Pi 5)
    i2c_addr : 7-bit I2C address of the Yahboom MCU (default 0x7A)
    """

    def __init__(self, i2c_bus: int = 1, i2c_addr: int = 0x7A) -> None:
        self._addr  = i2c_addr
        self._bus_n = i2c_bus
        self._lib   = None    # Raspbot library instance (if available)
        self._bus   = None    # smbus2 bus object (if available)
        self._sim   = False

        self._backend = self._init_backend()
        logger.info("[YahboomBoard] Backend: %s  addr=0x%02X  bus=%d",
                    self._backend, i2c_addr, i2c_bus)

    # ── Backend initialisation ────────────────────────────────────

    def _init_backend(self) -> str:
        import sys, os

        # Inject well-known Yahboom library paths so the official driver
        # is found even when it is not installed as a system package.
        _YAHBOOM_PATHS = [
            "/home/pi/py_install/Raspbot_Lib",
            "/home/pi/project_demo/raspbot",
            "/home/pi",
        ]
        for _p in _YAHBOOM_PATHS:
            if os.path.isdir(_p) and _p not in sys.path:
                sys.path.insert(0, _p)

        # 1. Try the official Yahboom library (Raspbot_Lib.py on RPi 5 image)
        try:
            from Raspbot_Lib import Raspbot  # type: ignore[import]
            self._lib = Raspbot()
            return "raspbot_lib"
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("[YahboomBoard] Raspbot_Lib load error: %s", exc)

        # 1b. Older image name
        try:
            from Raspbot import Raspbot          # type: ignore[import]
            self._lib = Raspbot()
            return "raspbot_lib"
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("[YahboomBoard] Raspbot lib load error: %s", exc)

        # 1c. Some images expose it as 'yahboom_Raspbot'
        try:
            from yahboom_Raspbot import Raspbot  # type: ignore[import]
            self._lib = Raspbot()
            return "yahboom_raspbot_lib"
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("[YahboomBoard] yahboom_Raspbot lib error: %s", exc)

        # 2. Direct smbus2 I2C (raw protocol)
        try:
            import smbus2                        # type: ignore[import]
            self._bus = smbus2.SMBus(self._bus_n)
            # Quick sanity-write: set all motors to stop
            self._i2c_set_motors(0, 0, 0, 0)
            return "smbus2_raw"
        except ImportError:
            logger.info("[YahboomBoard] smbus2 not found -> simulation")
        except Exception as exc:
            logger.warning("[YahboomBoard] smbus2 open failed (%s) -> simulation", exc)

        # 3. Simulation stub
        self._sim = True
        return "simulation"

    # ── Public motor API ─────────────────────────────────────────

    def set_motors(self, fl: int, fr: int, rl: int, rr: int) -> None:
        """
        Set individual wheel speeds.

        Parameters
        ----------
        fl, fr, rl, rr : duty cycle -100 … +100
                         positive = forward,  negative = reverse
        """
        fl = _clamp(fl)
        fr = _clamp(fr)
        rl = _clamp(rl)
        rr = _clamp(rr)

        if self._lib:
            # Yahboom official library — Ctrl_Muto(motor_id, speed -255..255)
            # IDs are 0-based: 0=L1(FL), 1=L2(RL), 2=R1(FR), 3=R2(RR)
            # Scale from our -100..+100 range to library's -255..+255
            try:
                for mid, val in enumerate([fl, rl, fr, rr], start=0):
                    scaled = int(val * 2.55)
                    self._lib.Ctrl_Muto(mid, scaled)
            except AttributeError:
                try:
                    self._lib.set_motor(fl, fr, rl, rr)
                except AttributeError:
                    self._lib.Ctrl_Motor(fl, fr, rl, rr)
        elif self._bus:
            self._i2c_set_motors(fl, fr, rl, rr)
        else:
            logger.debug("[Board-sim] motors FL=%d FR=%d RL=%d RR=%d",
                         fl, fr, rl, rr)

    def stop_motors(self) -> None:
        """Halt all wheels immediately."""
        self.set_motors(0, 0, 0, 0)

    # ── Public servo API ─────────────────────────────────────────

    def set_servo(self, servo_id: int, angle_deg: float) -> None:
        """
        Command a servo to an absolute angle.

        Parameters
        ----------
        servo_id  : 1 = pan (horizontal), 2 = tilt (vertical)
        angle_deg : target angle in degrees (0-180; 90 = center)
        """
        angle_deg = float(max(0.0, min(180.0, angle_deg)))

        if self._lib:
            try:
                self._lib.Ctrl_Servo(servo_id, int(angle_deg))
            except AttributeError:
                try:
                    self._lib.set_servo(servo_id, int(angle_deg))
                except AttributeError:
                    self._lib.ctrl_serve(servo_id, int(angle_deg))
        elif self._bus:
            self._i2c_set_servo(servo_id, angle_deg)
        else:
            logger.debug("[Board-sim] servo %d -> %.1f deg", servo_id, angle_deg)

    # ── Public ultrasonic API ─────────────────────────────────────

    def get_distance(self) -> float:
        """
        Trigger the HC-SR04 and return distance in centimetres.

        Returns the configured ``max_range_cm`` (default ~400 cm) if the
        sensor times out or returns an obviously-invalid reading.
        """
        if self._lib:
            try:
                # Trigger a measurement, then read hi/lo registers (in mm)
                # 0x1b = distance high byte, 0x1a = distance low byte
                # Must be read as separate calls - read_data_array(reg, 2)
                # does not return consecutive registers on this board.
                self._lib.Ctrl_Ulatist_Switch(1)
                time.sleep(0.08)
                hi = self._lib.read_data_array(0x1b, 1)[0]
                lo = self._lib.read_data_array(0x1a, 1)[0]
                raw_mm = (hi << 8) | lo
                # Empirical calibration: raw unit ~0.474 mm (not 0.5 mm).
                # Measured at known distances: actual = raw_mm / 9.5
                # (raw_mm / 20 * 2.11 ~ raw_mm / 9.48, rounded to 9.5)
                raw = raw_mm / 9.5  # cm
            except AttributeError:
                try:
                    raw = self._lib.Get_Distance()
                except AttributeError:
                    raw = self._lib.get_distance()
            except Exception:
                raw = None
            if raw is None or raw <= 0:
                return 400.0
            # Clamp garbage-high echoes (seen at close range) to max_range
            # so the blind-spot guard in ultrasonic.py can catch them.
            return min(float(raw), 400.0)
        elif self._bus:
            return self._i2c_get_distance()
        else:
            logger.debug("[Board-sim] get_distance -> 200.0 cm")
            return 200.0   # safe default in simulation

    # ── I2C helper methods ────────────────────────────────────────

    def _i2c_set_motors(self, fl: int, fr: int, rl: int, rr: int) -> None:
        """
        Write motor speeds over I2C.
        Encoding: -100…+100  →  0…200  (offset +100)
        """
        data = [
            _REG_MOTOR,
            fl + 100,
            fr + 100,
            rl + 100,
            rr + 100,
        ]
        try:
            self._bus.write_i2c_block_data(self._addr, data[0], data[1:])
        except Exception as exc:
            logger.warning("[YahboomBoard] I2C motor write failed: %s", exc)

    def _i2c_set_servo(self, servo_id: int, angle_deg: float) -> None:
        """
        Write servo position over I2C.
        Angle encoding: degrees × 10  → 0-1800 (two bytes, big-endian)
        """
        encoded = int(angle_deg * 10)  # 0-1800
        hi = (encoded >> 8) & 0xFF
        lo =  encoded       & 0xFF
        data = [servo_id, hi, lo]
        try:
            self._bus.write_i2c_block_data(self._addr, _REG_SERVO, data)
        except Exception as exc:
            logger.warning("[YahboomBoard] I2C servo write failed: %s", exc)

    def _i2c_get_distance(self) -> float:
        """
        Trigger ultrasonic and read distance over I2C.
        Returns cm; falls back to max_range on any I2C error.
        """
        try:
            # Trigger: write the register address to kick off a measurement
            self._bus.write_byte(self._addr, _REG_ULTRASONIC)
            time.sleep(_US_TRIGGER_WAIT_S)
            # Read 3 bytes: [hi, mid, lo] → distance in mm
            raw = self._bus.read_i2c_block_data(self._addr, _REG_ULTRASONIC, 3)
            dist_mm = (raw[0] << 16) | (raw[1] << 8) | raw[2]
            dist_cm = dist_mm / 10.0
            if dist_cm <= 0 or dist_cm > 500:
                return 400.0
            return dist_cm
        except Exception as exc:
            logger.debug("[YahboomBoard] I2C distance read failed: %s", exc)
            return 400.0

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        """Release hardware resources."""
        try:
            self.stop_motors()
        except Exception:
            pass
        if self._bus:
            try:
                self._bus.close()
            except Exception:
                pass
        logger.debug("[YahboomBoard] Closed")


# ── Helpers ───────────────────────────────────────────────────────

def _clamp(v: int, lo: int = -100, hi: int = 100) -> int:
    return max(lo, min(hi, int(v)))
