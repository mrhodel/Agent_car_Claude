"""
rl/environment.py
Gym-compatible wrapper that turns the robot hardware + perception +
mapping stack into a reinforcement learning environment.

Supports two modes:
  • "robot"  – runs entirely on live hardware (RPi 5)
  • "sim"    – fully simulated, no hardware required (for training)

State vector composition (all float32):
  ┌─────────────────────────────────────────────────────┐
  │ us_forward        (1)          normalised 0-1       │
  │ visual_features   (128)        from MobileNetV2     │
  │ depth_map_flat    (256 = 16×16)normalised 0-1       │
  │ local_map_flat    (49  = 7×7)  occupancy probs      │
  │ sin(theta), cos(theta)         heading              │
  │ vx, vy, omega                  current velocity     │
  └─────────────────────────────────────────────────────┘
Total = 1 + 128 + 256 + 49 + 2 + 3 = 439

Note: ultrasonic_rays=1 because the HC-SR04 is a FIXED sensor that
always points forward.  Only the camera (on the gimbal) can sweep.

Action space: Discrete(11)
  0  forward          5  rotate_right
  1  backward         6  gimbal_pan_left
  2  strafe_left      7  gimbal_pan_right
  3  strafe_right     8  gimbal_tilt_up
  4  rotate_left      9  gimbal_tilt_down
                     10  stop
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Action index constants for readability
ACT_FORWARD       = 0
ACT_BACKWARD      = 1
ACT_STRAFE_LEFT   = 2
ACT_STRAFE_RIGHT  = 3
ACT_ROTATE_LEFT   = 4
ACT_ROTATE_RIGHT  = 5
ACT_PAN_LEFT      = 6
ACT_PAN_RIGHT     = 7
ACT_TILT_UP       = 8
ACT_TILT_DOWN     = 9
ACT_STOP          = 10
N_ACTIONS         = 11


class RobotEnv:
    """
    RL environment wrapping the full robot stack.

    Parameters
    ----------
    cfg        : full config dict (from robot_config.yaml)
    motors     : MotorController instance
    ultrasonic : UltrasonicSensor instance
    gimbal     : Gimbal instance
    camera     : Camera instance
    vision     : VisionPipeline instance
    grid       : OccupancyGrid instance
    controller : MotionController instance
    mode       : "robot" or "sim"
    """

    def __init__(
        self,
        cfg: dict,
        motors=None,
        ultrasonic=None,
        gimbal=None,
        camera=None,
        vision=None,
        grid=None,
        controller=None,
        mode: str = "robot",
    ) -> None:
        self._cfg        = cfg
        self._motors     = motors
        self._us         = ultrasonic
        self._gimbal     = gimbal
        self._camera     = camera
        self._vision     = vision
        self._grid       = grid
        self._controller = controller
        self._mode       = mode

        rl_cfg = cfg.get("rl", {})
        state_cfg   = rl_cfg.get("state",   {})
        action_cfg  = rl_cfg.get("actions", {})

        self._n_rays        = int(state_cfg.get("ultrasonic_rays",    1))  # fixed sensor = 1
        self._feat_dim      = int(state_cfg.get("visual_feature_dim", 128))
        self._depth_flat    = int(state_cfg.get("depth_map_flat",     256))
        self._map_size      = int(state_cfg.get("local_map_size",     7))
        self._include_vel   = bool(state_cfg.get("include_velocity",  True))
        self._include_hdg   = bool(state_cfg.get("include_heading",   True))

        self._base_speed    = int(action_cfg.get("base_speed",    55))
        self._strafe_speed  = int(action_cfg.get("strafe_speed",  30))
        self._reverse_speed = int(action_cfg.get("reverse_speed", 30))
        self._rotate_speed  = int(action_cfg.get("rotate_speed",  45))
        self._gimbal_step   = float(action_cfg.get("gimbal_step_deg", 10))
        self._side_depth_close = float(action_cfg.get("side_depth_close", 0.75))
        # Cache last depth map for side clearance checks
        self._last_depth_map: np.ndarray = np.full((16, 16), 0.5, dtype=np.float32)
        # No ray_angles needed: ultrasonic is fixed (always forward = 0°)

        self._emergency_cm  = float(cfg.get("agent", {}).get(
                                  "safety", {}).get("emergency_stop_cm", 10))

        # State size
        extra = 0
        if self._include_vel: extra += 3   # vx, vy, omega
        if self._include_hdg: extra += 2   # sin, cos
        self.state_size = (self._n_rays + self._feat_dim +
                           self._depth_flat + self._map_size**2 + extra)
        self.n_actions  = N_ACTIONS

        # Internal state
        self._robot_x     = 0.0
        self._robot_y     = 0.0
        self._robot_theta = 0.0
        self._vx = self._vy = self._omega = 0.0
        self._step_count  = 0
        self._max_steps   = int(rl_cfg.get("ppo", {}).get("max_steps_per_episode", 500))
        self._consecutive_backward = 0   # stuck-backing-into-wall detector
        self._last_us_cm           = 400.0

        # For simulation mode
        self._sim_obstacles: list = []   # list of (x, y, radius_m)
        self._sim_init_obstacles()

        logger.info("[Env] mode=%s  state_dim=%d  actions=%d",
                    mode, self.state_size, N_ACTIONS)

    # ── Gym interface ─────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self._robot_x     = 0.0
        self._robot_y     = 0.0
        self._robot_theta = 0.0
        self._vx = self._vy = self._omega = 0.0
        self._step_count            = 0
        self._prev_action           = -1
        self._consecutive_rotations = 0
        self._consecutive_backward  = 0
        self._last_us_cm            = 400.0

        if self._mode == "robot":
            if self._motors: self._motors.stop()
            if self._gimbal: self._gimbal.centre()
            # Wall escape: back away from front obstacle, then nudge forward
            # to clear any rear wall the robot may have backed into.
            if self._us and self._motors:
                try:
                    for attempt in range(5):
                        dist = self._us.read_cm()
                        if dist >= 25.0:
                            break
                        logger.info("[Env] Wall escape attempt %d: dist=%.1f cm",
                                    attempt + 1, dist)
                        self._motors.move_backward(40)
                        time.sleep(0.4)
                        self._motors.stop()
                        time.sleep(0.15)
                    # Always nudge forward to clear any rear wall
                    logger.debug("[Env] Reset forward nudge")
                    self._motors.move_forward(40)
                    time.sleep(0.3)
                finally:
                    self._motors.stop()
            time.sleep(0.1)
        else:
            self._sim_place_robot()

        if self._grid:
            # Reset grid state in-place (no re-allocation)
            self._grid.soft_reset()

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply action, observe result.

        Returns
        -------
        state     : np.ndarray
        reward    : float
        done      : bool
        info      : dict
        """
        prev_explored = self._grid.explored_cells if self._grid else 0

        # Execute action
        collided = self._execute_action(action)

        # Progress pose estimate (dead-reckoning / sim)
        self._update_pose(action)

        # Sense
        us_dists, obs_result = self._sense()

        # Map update – ultrasonic is single forward reading
        new_cells = 0
        if self._grid:
            from mapping.occupancy_grid import RobotPose
            pose = RobotPose(self._robot_x, self._robot_y, self._robot_theta)
            heading_deg = math.degrees(self._robot_theta)
            us_dist = us_dists[0] if us_dists else 400.0
            new_cells = self._grid.update(
                pose,
                obstacle_angles_deg=[heading_deg] if us_dist < 300 else [],
                obstacle_dists_cm=[us_dist] if us_dist < 300 else [],
                free_angles_deg=[heading_deg] if us_dist >= 300 else [],
                free_dists_cm=[us_dist] if us_dist >= 300 else [],
            )

        nearest_cm = min(us_dists) if us_dists else 200.0

        # Reward
        from rl.reward import RewardCalculator, _SPINNING_ACTIONS
        rew_cfg = self._cfg.get("rl", {}).get("reward", {})
        prev_action = getattr(self, "_prev_action", action)
        if action in _SPINNING_ACTIONS:
            self._consecutive_rotations = getattr(self, "_consecutive_rotations", 0) + 1
        else:
            self._consecutive_rotations = 0
        reward = RewardCalculator.compute(
            new_explored_cells=new_cells,
            nearest_obstacle_cm=nearest_cm,
            collided=collided,
            action=action,
            prev_action=prev_action,
            cfg=rew_cfg,
            consecutive_rotations=self._consecutive_rotations,
        )
        self._prev_action = action

        # ── Stuck-backward detector ───────────────────────────────
        # If the robot keeps choosing backward and the sensor reading
        # barely changes (rear wall), the episode is unrecoverable.
        # Force done so reset() clears it with a forward nudge.
        if action == ACT_BACKWARD:
            us_now = us_dists[0] if us_dists else 400.0
            if abs(us_now - self._last_us_cm) < 3.0:   # sensor not moving
                self._consecutive_backward += 1
            else:
                self._consecutive_backward = 0
            self._last_us_cm = us_now
        else:
            self._consecutive_backward = 0

        self._step_count += 1
        # Emergency stop triggers on distance alone (collided is not reliable
        # in robot mode because _execute_action_robot always returns False)
        emergency = nearest_cm < self._emergency_cm
        stuck_backward = (self._consecutive_backward >= 6
                          and self._mode == "robot")
        if stuck_backward:
            logger.warning("[Env] Stuck backing into wall after %d steps — forcing reset",
                           self._consecutive_backward)
        done = (emergency or collided or stuck_backward
                or self._step_count >= self._max_steps)
        if (emergency or stuck_backward) and self._mode == "robot" and self._motors:
            self._motors.stop()

        state = self._get_state(obs_result, us_dists)
        info  = {
            "step": self._step_count,
            "new_cells": new_cells,
            "nearest_cm": nearest_cm,
            "collided": collided,
            "stuck_backward": stuck_backward,
        }
        return state, reward, done, info

    # ── State assembly ────────────────────────────────────────────

    def _get_state(self, obs_result=None, us_dists=None) -> np.ndarray:
        parts = []

        # Ultrasonic rays (normalised)
        if us_dists is not None and len(us_dists) == self._n_rays:
            parts.append(np.array(us_dists, dtype=np.float32) / 400.0)
        else:
            parts.append(np.full(self._n_rays, 0.5, dtype=np.float32))

        # Visual features
        if obs_result is not None:
            parts.append(np.array(obs_result.visual_features, dtype=np.float32)[:self._feat_dim])
        else:
            parts.append(np.zeros(self._feat_dim, dtype=np.float32))

        # Depth map (flat) — use cached map on vision-skipped steps
        dm_src = obs_result.depth_map if obs_result is not None else self._last_depth_map
        dm = dm_src.flatten()[:self._depth_flat]
        dm_pad = np.zeros(self._depth_flat, dtype=np.float32)
        dm_pad[:len(dm)] = dm
        parts.append(dm_pad)

        # Local map window
        if self._grid:
            from mapping.occupancy_grid import RobotPose
            pose = RobotPose(self._robot_x, self._robot_y, self._robot_theta)
            win = self._grid.get_local_window(pose, self._map_size)
            parts.append(win.flatten())
        else:
            parts.append(np.full(self._map_size**2, 0.5, dtype=np.float32))

        # Heading
        if self._include_hdg:
            parts.append(np.array([math.sin(self._robot_theta),
                                    math.cos(self._robot_theta)],
                                   dtype=np.float32))

        # Velocity
        if self._include_vel:
            parts.append(np.array([self._vx, self._vy, self._omega],
                                   dtype=np.float32))

        state = np.concatenate(parts)
        # Safety: clip and handle NaN
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        state = np.clip(state, -5.0, 5.0)
        return state.astype(np.float32)

    # ── Action execution ──────────────────────────────────────────

    def _execute_action(self, action: int) -> bool:
        """Apply action; return True if collision detected."""
        if self._mode == "robot":
            return self._execute_action_robot(action)
        return self._execute_action_sim(action)

    def _execute_action_robot(self, action: int) -> bool:
        # ── Normal action execution ───────────────────────────────
        if self._controller:
            self._controller.move_action(action, self._base_speed,
                                          self._rotate_speed,
                                          self._strafe_speed,
                                          self._reverse_speed)
        if self._gimbal:
            step = self._gimbal_step
            if action == ACT_PAN_LEFT:
                self._gimbal.move_pan(-step)
            elif action == ACT_PAN_RIGHT:
                self._gimbal.move_pan( step)
            elif action == ACT_TILT_UP:
                self._gimbal.move_tilt( step)
            elif action == ACT_TILT_DOWN:
                self._gimbal.move_tilt(-step)
        # Brief execution time — always stop motors even on interrupt
        try:
            time.sleep(0.08)
        finally:
            if self._motors and action != ACT_STOP:
                self._motors.stop()
        return False   # collision detection is done via sensor readings

    def _execute_action_sim(self, action: int) -> bool:
        """Simulate motion kinematics."""
        dt   = 0.1    # step duration (s)
        spd  = 0.3    # m/s at base speed
        rot  = 0.8    # rad/s for rotation
        dx = dy = dtheta = 0.0
        if action == ACT_FORWARD:     dx = spd * dt
        elif action == ACT_BACKWARD:  dx = -spd * dt
        elif action == ACT_STRAFE_LEFT:  dy = -spd * dt
        elif action == ACT_STRAFE_RIGHT: dy =  spd * dt
        elif action == ACT_ROTATE_LEFT:  dtheta =  rot * dt
        elif action == ACT_ROTATE_RIGHT: dtheta = -rot * dt

        # Rotate displacement into world frame
        ct = math.cos(self._robot_theta)
        st = math.sin(self._robot_theta)
        wx = dx * ct - dy * st
        wy = dx * st + dy * ct
        nx = self._robot_x + wx
        ny = self._robot_y + wy
        ntheta = self._robot_theta + dtheta

        # Collision check in simulation
        for ox, oy, radius in self._sim_obstacles:
            if math.hypot(nx - ox, ny - oy) < radius + 0.18:
                return True   # collision
        self._robot_x = nx
        self._robot_y = ny
        self._robot_theta = ntheta
        self._vx  = wx / dt
        self._vy  = wy / dt
        self._omega = dtheta / dt
        return False

    def _update_pose(self, action: int) -> None:
        """For robot mode: integrate wheel odometry (stub – can be extended)."""
        # TODO: integrate encoder feedback when available
        pass

    # ── Sensing ───────────────────────────────────────────────────

    def _sense(self):
        """Collect sensor readings and run vision pipeline.
        
        Vision (MiDaS depth) is expensive on Pi CPU (~600ms). We run it every
        `vision_every_n_steps` steps and reuse the cached depth map otherwise.
        """
        # Ultrasonic: single fixed forward reading (sensor does NOT sweep)
        if self._mode == "robot" and self._us:
            us_dist = self._us.read_cm()
        elif self._mode == "sim":
            us_dist = self._sim_us_forward()
        else:
            us_dist = 200.0

        us_dists = [us_dist]   # 1-element list; shape matches self._n_rays == 1

        obs_result = None
        vision_interval = int(self._cfg.get("rl", {}).get("vision_every_n_steps", 3))
        if self._camera and self._vision:
            if (self._step_count % vision_interval) == 0:
                frame = self._camera.read()
                if frame is not None:
                    obs_result = self._vision.process(frame, us_dist)
                    if obs_result is not None and obs_result.depth_map is not None:
                        self._last_depth_map = obs_result.depth_map

        return us_dists, obs_result

    # ── Simulation helpers ────────────────────────────────────────

    def _sim_init_obstacles(self) -> None:
        """Generate a simple random indoor room layout."""
        rng = np.random.default_rng(0)
        n = 12
        xs = rng.uniform(-4, 4, n)
        ys = rng.uniform(-4, 4, n)
        rs = rng.uniform(0.15, 0.40, n)
        self._sim_obstacles = list(zip(xs.tolist(), ys.tolist(), rs.tolist()))
        # Hard walls at ±5 m
        wall_pts = [
            (-5, 0, 0.1), (5, 0, 0.1),
            (0, -5, 0.1), (0, 5, 0.1),
        ]
        self._sim_obstacles.extend(wall_pts)

    def _sim_place_robot(self) -> None:
        """Place robot in a free location at start of episode."""
        rng = np.random.default_rng(int(time.time() * 1000) % 2**31)
        for _ in range(100):
            x = rng.uniform(-3, 3)
            y = rng.uniform(-3, 3)
            if all(math.hypot(x-ox, y-oy) > 0.4
                   for ox, oy, _ in self._sim_obstacles):
                self._robot_x = x
                self._robot_y = y
                self._robot_theta = rng.uniform(-math.pi, math.pi)
                return
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_theta = 0.0

    def _sim_us_forward(self) -> float:
        """
        Single forward ray-cast for the fixed ultrasonic in simulation.
        The sensor always fires straight ahead (robot heading).
        """
        angle_world = self._robot_theta  # radians, forward direction
        min_dist = 400.0
        for ox, oy, radius in self._sim_obstacles:
            dx = ox - self._robot_x
            dy = oy - self._robot_y
            proj = dx * math.cos(angle_world) + dy * math.sin(angle_world)
            if proj <= 0:
                continue
            perp2 = (dx**2 + dy**2) - proj**2
            if perp2 > radius**2:
                continue
            hit = proj - math.sqrt(max(0, radius**2 - perp2))
            dist_cm = hit * 100.0
            if 2 <= dist_cm < min_dist:
                min_dist = dist_cm
        return min_dist

    # Keep legacy name as alias for backward-compatibility in subclasses
    def _sim_us_rays(self) -> list:
        """Deprecated: use _sim_us_forward().  Returns 1-element list."""
        return [self._sim_us_forward()]
