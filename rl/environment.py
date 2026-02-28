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
        self._last_obs_result = None
        self._need_frame_after_t: float = 0.0
        self._force_vision_next: bool = False
        self._pre_gimbal_frame = None
        self._no_fresh_vision_steps = 0
        self._last_failsafe_warn_t = 0.0
        # No ray_angles needed: ultrasonic is fixed (always forward = 0°)

        self._emergency_cm  = float(cfg.get("agent", {}).get(
                                  "safety", {}).get("emergency_stop_cm", 10))
        self._min_range_cm = float(cfg.get("hal", {}).get(
                                  "ultrasonic", {}).get("min_range_cm", 3.0))

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
                # min_range reading (e.g. 3.0 cm) means the HC-SR04 blind spot
                # fired — the sensor physically cannot measure <7 cm and returns
                # min_range as a sentinel.  Treat it as "unknown", not a
                # confirmed obstacle, so we only escape on readings that are
                # clearly in range AND consistently low.
                _min_r = float(self._cfg.get("hal", {}).get(
                               "ultrasonic", {}).get("min_range_cm", 3.0))
                try:
                    for attempt in range(5):
                        d1 = self._us.read_cm()
                        time.sleep(0.05)
                        d2 = self._us.read_cm()
                        # Skip if either reading is exactly min_range (blind spot)
                        if d1 <= _min_r or d2 <= _min_r:
                            logger.debug("[Env] Wall escape suppressed: blind-spot"
                                         " reading (d1=%.1f d2=%.1f cm)", d1, d2)
                            break
                        if d1 >= 25.0 or d2 >= 25.0:
                            if d1 < 25.0 or d2 < 25.0:
                                logger.debug("[Env] Spurious US reading suppressed"
                                             " at reset (d1=%.1f d2=%.1f cm)", d1, d2)
                            break
                        # Both readings confirmed in-range and low — real obstacle
                        dist = max(d1, d2)
                        logger.info("[Env] Wall escape attempt %d: dist=%.1f cm",
                                    attempt + 1, dist)
                        self._motors.move_backward(40)
                        time.sleep(0.4)
                        self._motors.stop()
                        time.sleep(0.15)
                    # ── SAFETY CHECK ──
                    # Ensure we are not starting the episode inside the emergency radius.
                    # If we are too close (< 25cm), force a backward move.
                    
                    _safe_start_cm = 25.0
                    for _bk in range(3):
                        _d_check = self._us.read_cm()
                        if _d_check > _safe_start_cm:
                            break
                        
                        logger.warning("[Env] Too close at start (%.1f cm) -> Backing up...", _d_check)
                        self._motors.move_backward(45)
                        time.sleep(0.4)
                        self._motors.stop()
                        time.sleep(0.2)

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

        # nearest_cm: use raw min.  HC-SR04 returns min_range_cm (3.0) when the
        # target is in the sensor blind spot (<~7 cm) — that IS a genuine close-
        # range event and should trigger emergency stop just as a real reading
        # would.  Do NOT filter it out.
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

                
        done_reason = "active"
        if emergency:       done_reason = "emergency_stop"
        elif collided:      done_reason = "collision"
        elif stuck_backward: done_reason = "stuck_backward"
        elif self._step_count >= self._max_steps: done_reason = "timeout"

        state = self._get_state(obs_result, us_dists)
        info  = {
            "step": self._step_count,
            "new_cells": new_cells,
            "nearest_cm": nearest_cm,
            "collided": collided,
            "stuck_backward": stuck_backward,
            "done_reason": done_reason,
        }
        if self._cfg.get("render_sim", False) and self._mode == "sim": self.render()
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
        # FAILSAFE_GUARD_MOTION: do not drive while awaiting a verified post-gimbal view
        if action in (ACT_FORWARD, ACT_BACKWARD, ACT_STRAFE_LEFT, ACT_STRAFE_RIGHT, ACT_ROTATE_LEFT, ACT_ROTATE_RIGHT):
            if self._need_frame_after_t > 0.0 or bool(getattr(self, '_force_vision_next', False)): 
                if self._motors: self._motors.stop()
                now = time.monotonic()
                if (now - self._last_failsafe_warn_t) > 1.0:
                    logger.warning('[FailSafe] Blocking motion until post-pan fresh vision is available')
                    self._last_failsafe_warn_t = now
                return False
            # Also refuse motion if camera frames are too old (don't drive blind).
            if self._camera and hasattr(self._camera, 'read_with_timestamp'):
                _f, _t = self._camera.read_with_timestamp()
                if not _t or (time.monotonic() - float(_t)) > 1.0:
                    if self._motors: self._motors.stop()
                    self._force_vision_next = True
                    self._no_fresh_vision_steps += 1
                    now = time.monotonic()
                    if (now - self._last_failsafe_warn_t) > 1.0:
                        logger.warning('[FailSafe] Blocking motion: camera frames stale (>1.0s)')
                        self._last_failsafe_warn_t = now
                    return False
        # ── Normal action execution ───────────────────────────────
        if self._controller:
            self._controller.move_action(action, self._base_speed,
                                          self._rotate_speed,
                                          self._strafe_speed,
                                          self._reverse_speed)
        if self._gimbal:
            step = self._gimbal_step
            if action == ACT_PAN_LEFT:
                if self._camera:
                    try:
                        self._pre_gimbal_frame = self._camera.read()
                    except Exception:
                        self._pre_gimbal_frame = None
                self._gimbal.move_pan(-step)
                self._need_frame_after_t = time.monotonic()
                self._force_vision_next = True
            elif action == ACT_PAN_RIGHT:
                if self._camera:
                    try:
                        self._pre_gimbal_frame = self._camera.read()
                    except Exception:
                        self._pre_gimbal_frame = None
                self._gimbal.move_pan( step)
                self._need_frame_after_t = time.monotonic()
                self._force_vision_next = True
            elif action == ACT_TILT_UP:
                if self._camera:
                    try:
                        self._pre_gimbal_frame = self._camera.read()
                    except Exception:
                        self._pre_gimbal_frame = None
                self._gimbal.move_tilt( step)
                self._need_frame_after_t = time.monotonic()
                self._force_vision_next = True
            elif action == ACT_TILT_DOWN:
                if self._camera:
                    try:
                        self._pre_gimbal_frame = self._camera.read()
                    except Exception:
                        self._pre_gimbal_frame = None
                self._gimbal.move_tilt(-step)
                self._need_frame_after_t = time.monotonic()
                self._force_vision_next = True
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
        
        Vision (MiDaS depth / YOLO / feature extraction) is expensive on Pi CPU.
        We run it every `vision_every_n_steps` steps and reuse cached results
        otherwise.
        """
        # Ultrasonic: single fixed forward reading (sensor does NOT sweep)
        if self._mode == "robot" and self._us:
            us_dist = self._us.read_cm()
        elif self._mode == "sim":
            us_dist = self._sim_us_forward()
        else:
            us_dist = 200.0

        us_dists = [us_dist]   # 1-element list; shape matches self._n_rays == 1
        obs_result = self._last_obs_result

        rl_cfg = self._cfg.get("rl", {})
        # Support both legacy and current config placement.
        vision_interval = (
            rl_cfg.get("vision_every_n_steps")
            if rl_cfg.get("vision_every_n_steps") is not None
            else rl_cfg.get("ppo", {}).get("vision_every_n_steps")
        )
        try:
            vision_interval = int(vision_interval) if vision_interval is not None else 3
        except Exception:
            vision_interval = 3
        vision_interval = max(1, vision_interval)

        force_vision = bool(self._force_vision_next)
        self._force_vision_next = False
        should_run_vision = force_vision or ((self._step_count % vision_interval) == 0)

        def _mad(a, b) -> float:
            try:
                import numpy as _np
                if a is None or b is None: return 1e9
                if a.shape != b.shape: return 1e9
                return float(_np.mean(_np.abs(a.astype(_np.int16) - b.astype(_np.int16))))
            except Exception:
                return 1e9

        if self._mode == "sim" and self._vision and should_run_vision:
            frame = self._render_camera_sim()
            if frame is not None:
                new_result = self._vision.process(frame, us_dist)
                if new_result is not None:
                    obs_result = new_result
                    self._last_obs_result = new_result
                    if new_result.depth_map is not None:
                        self._last_depth_map = new_result.depth_map

        if self._camera and self._vision and should_run_vision:
            frame = None
            if self._need_frame_after_t > 0.0:
                after_t = float(self._need_frame_after_t)
                frame = self._camera.read_fresh(after_t=after_t, timeout_s=0.6)
                # MAD_THRESHOLD_AFTER_GIMBAL: ensure the view actually changed post-move
                if frame is not None and self._pre_gimbal_frame is not None:
                    if _mad(frame, self._pre_gimbal_frame) < 0.3:
                        # Likely still the pre-move buffer (or camera stuck). Treat as not fresh.
                        frame = None

                if frame is None:
                    # Keep trying next step until we get a post-move frame.
                    self._force_vision_next = True
                    self._no_fresh_vision_steps += 1
                    # MAX_NO_FRESH_STEPS
                    if self._no_fresh_vision_steps >= 20:
                        if self._motors: self._motors.stop()
                        logger.error('[FailSafe] Vision did not update after gimbal move for 20 steps; holding STOP')

                    logger.warning(
                        '[Env] No fresh frame after gimbal move (timeout=%.2fs) — using cached observation',
                        0.6,
                    )
                else:
                    self._need_frame_after_t = 0.0
                    self._pre_gimbal_frame = None
                    self._no_fresh_vision_steps = 0
            else:
                frame = self._camera.read()

            if frame is not None:
                new_result = self._vision.process(frame, us_dist)
                if new_result is not None:
                    obs_result = new_result
                    self._last_obs_result = new_result
                    if new_result.depth_map is not None:
                        self._last_depth_map = new_result.depth_map

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

    def render(self) -> None:
        """Visualize simulation state using OpenCV."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return

        # 10m x 10m area (-5..5), 500x500 img => 50 px/m
        width = 500
        mid = width // 2
        scale = 50.0

        # White background
        img = np.full((width, width, 3), 255, dtype=np.uint8)

        # Draw obstacles (grey)
        for ox, oy, r in self._sim_obstacles:
            # Transform to image coords (x right, y up in sim -> y down in img)
            cx = int(mid + ox * scale)
            cy = int(mid - oy * scale)
            cr = int(r * scale)
            cv2.circle(img, (cx, cy), cr, (100, 100, 100), -1)

        # Draw robot
        rx = int(mid + self._robot_x * scale)
        ry = int(mid - self._robot_y * scale)
        
        # Body (Blue)
        cv2.circle(img, (rx, ry), 8, (255, 0, 0), -1)
        
        # Heading (Red Line) - show orientation
        # Robot theta 0 is East (Right), pi/2 is North (Up in Sim, Down in Img?? wait)
        # Math: y axis is usually up. Screen y is down.
        # So sim(y) -> img(mid - y).
        # sin(theta) affects y. if theta=pi/2, y increases. img_y decreases. 
        # So we subtract sin component. Correct.
        
        # Sensor ray (Green)
        dist_m = self._sim_us_forward() / 100.0
        draw_dist = min(dist_m, 2.0)
        
        ex = int(rx + (draw_dist * scale) * math.cos(self._robot_theta))
        ey = int(ry - (draw_dist * scale) * math.sin(self._robot_theta))
        
        cv2.line(img, (rx, ry), (ex, ey), (0, 255, 0), 1)

        # Show window
        cv2.imshow("Robot Sim (10x10m Room)", img)
        cv2.waitKey(1)

    def _render_camera_sim(self) -> np.ndarray:
        """Render a simulated first-person view of the room for vision training."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None

        # Camera view parameters
        width, height = 320, 240
        fov_deg = 60.0
        max_dist = 5.0
        
        # White background (walls/void)
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        
        # Draw floor (light grey) - horizon is halfway
        cv2.rectangle(img, (0, height//2), (width, height), (240, 240, 240), -1)

        # Simple ray-casting for obstacles
        # We cast rays across the FOV to find distance to obstacles
        n_rays = width // 4  # low-res raycast for speed
        angle_step = math.radians(fov_deg) / n_rays
        start_angle = self._robot_theta + math.radians(fov_deg/2)
        
        for i in range(n_rays):
            angle = start_angle - i * angle_step
            # Find closest intersection
            min_dist = max_dist
            
            for ox, oy, r in self._sim_obstacles:
                # Math similar to _sim_us_forward but for arbitary angle
                dx = ox - self._robot_x
                dy = oy - self._robot_y
                proj = dx * math.cos(angle) + dy * math.sin(angle)
                if proj <= 0: continue
                perp2 = (dx**2 + dy**2) - proj**2
                if perp2 > r**2: continue
                hit = proj - math.sqrt(max(0, r**2 - perp2))
                if hit < min_dist:
                    min_dist = hit
            
            if min_dist < max_dist:
                # Draw vertical strip for obstacle
                # Height is inversely proportional to distance
                h = int(height / (min_dist + 0.1))
                h = min(h, height)
                x_start = i * 4
                x_end = (i+1) * 4
                color = int(255 * (min_dist / max_dist)) # Darker when close
                color = max(0, min(255, color))
                # Draw "column"
                cv2.rectangle(img, (x_start, height//2 - h//2), 
                                   (x_end, height//2 + h//2), 
                                   (color, color, color), -1)
                                   
        return img
