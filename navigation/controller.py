"""
navigation/controller.py
Pure-pursuit motion controller for the Mecanum-wheeled Raspbot V2.

Tracks a planned Path by:
  1. Finding the lookahead point on the remaining path.
  2. Computing a desired heading toward it.
  3. Decomposing into vx, vy, omega for the Mecanum IK.

Recovery behaviours (spin, backup) are also managed here.
"""
from __future__ import annotations

import logging
import math
import time
from typing import List, Optional, Tuple

from hal.motors import MotorController
from navigation.path_planner import Path

logger = logging.getLogger(__name__)


class MotionController:
    """
    Parameters
    ----------
    motors : MotorController
    cfg    : dict – the ``navigation.controller`` sub-dict
    """

    def __init__(self, motors: MotorController, cfg: dict) -> None:
        self._motors         = motors
        self._lookahead      = float(cfg.get("lookahead_dist_m",  0.20))
        self._goal_tol       = float(cfg.get("goal_tolerance_m",  0.10))
        self._angle_kp       = float(cfg.get("angle_kp",          1.2))
        self._linear_kp      = float(cfg.get("linear_kp",         0.8))
        self._max_ang_vel    = float(cfg.get("max_angular_vel",    60))
        self._max_lin_vel    = float(cfg.get("max_linear_vel",     70))
        self._rec_cfg        = {}   # filled by set_recovery_config()

        # Recovery state
        self._recovery_attempts = 0
        self._max_recovery      = 5

    def set_recovery_config(self, cfg: dict) -> None:
        self._rec_cfg = cfg
        self._max_recovery = int(cfg.get("max_attempts", 5))

    # ── Path following ────────────────────────────────────────────

    def follow_step(
        self,
        path: Path,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        nearest_obstacle_cm: float,
    ) -> bool:
        """
        Execute one control step along the path.

        Parameters
        ----------
        path                : current planned path
        robot_x/y           : current world position (metres)
        robot_theta         : current heading (radians)
        nearest_obstacle_cm : from sensor fusion (safety override)

        Returns
        -------
        True if goal reached, False otherwise.
        """
        if path.empty:
            self._motors.stop()
            return True

        # Find lookahead point
        target = self._lookahead_point(path.waypoints, robot_x, robot_y)
        if target is None:
            self._motors.stop()
            return True

        tx, ty = target
        dx = tx - robot_x
        dy = ty - robot_y
        dist = math.hypot(dx, dy)

        # Goal check (last waypoint)
        gx, gy = path.waypoints[-1]
        if math.hypot(gx - robot_x, gy - robot_y) < self._goal_tol:
            self._motors.stop()
            logger.debug("[Controller] Goal reached")
            return True

        # Desired heading
        desired_heading = math.atan2(dy, dx)
        heading_error   = _angle_wrap(desired_heading - robot_theta)

        # Mecanum: project forward velocity into body frame
        vx = math.cos(heading_error) * dist * self._linear_kp
        vy = math.sin(heading_error) * dist * self._linear_kp
        omega = heading_error * self._angle_kp

        # Clamp to limits
        lin_speed = math.hypot(vx, vy)
        if lin_speed > 1.0:
            vx /= lin_speed; vy /= lin_speed
        vx    = max(-1.0, min(1.0, vx))
        vy    = max(-1.0, min(1.0, vy))
        omega = max(-1.0, min(1.0, omega / math.pi))

        self._motors.set_velocity(vx, vy, omega)
        return False

    # ── Point-to-point move (used by RL discrete actions) ─────────

    def move_action(self, action: int, speed: int = 55,
                    rotate_speed: int = 45,
                    strafe_speed: int = 30,
                    reverse_speed: int = 30) -> None:
        """
        Map a discrete RL action index to a motor command.

        Actions: 0=forward, 1=backward, 2=strafe_left, 3=strafe_right,
                 4=rotate_left, 5=rotate_right, 10=stop.
        """
        if action == 0:
            self._motors.move_forward(speed)
        elif action == 1:
            self._motors.move_backward(reverse_speed)
        elif action == 2:
            self._motors.strafe_left(strafe_speed)
        elif action == 3:
            self._motors.strafe_right(strafe_speed)
        elif action == 4:
            self._motors.rotate_left(rotate_speed)
        elif action == 5:
            self._motors.rotate_right(rotate_speed)
        else:
            self._motors.stop()

    # ── Recovery ─────────────────────────────────────────────────

    def recover(self) -> bool:
        """
        Execute a recovery manoeuvre.
        Returns True if recovery exhausted (give up), False otherwise.
        """
        if self._recovery_attempts >= self._max_recovery:
            logger.warning("[Controller] Recovery exhausted")
            self._motors.stop()
            return True

        spin_deg  = float(self._rec_cfg.get("spin_degrees",     30))
        back_dur  = float(self._rec_cfg.get("backup_duration_s", 0.5))

        logger.info("[Controller] Recovery attempt %d/%d",
                    self._recovery_attempts + 1, self._max_recovery)

        # Backup briefly
        self._motors.move_backward(40)
        time.sleep(back_dur)

        # Spin to find a clear direction
        self._motors.rotate_left(40)
        # Approximate spin: 30 deg at ~40 duty speed ≈ 0.5 s on carpet
        est_spin_time = (spin_deg / 90.0) * 0.8
        time.sleep(est_spin_time)

        self._motors.stop()
        self._recovery_attempts += 1
        return False

    def reset_recovery(self) -> None:
        self._recovery_attempts = 0

    # ── Internal ─────────────────────────────────────────────────

    def _lookahead_point(
        self,
        waypoints: List[Tuple[float, float]],
        rx: float,
        ry: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Find the first waypoint at least ``lookahead`` distance from robot.
        """
        for wx, wy in reversed(waypoints):
            if math.hypot(wx - rx, wy - ry) >= self._lookahead:
                return wx, wy
        # All waypoints inside lookahead → aim for the last one
        return waypoints[-1] if waypoints else None


def _angle_wrap(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a
