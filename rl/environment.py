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
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional, Tuple, List
from types import SimpleNamespace

import numpy as np

# Try to import typical perception structs, or fallback for sim-only environments
try:
    from perception.vision import PerceptionResult
except ImportError:
    PerceptionResult = None

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

        self._n_rays        = int(state_cfg.get("ultrasonic_rays",    1))
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
        
        # Cache last depth map for side clearance checks
        self._last_depth_map: np.ndarray = np.full((16, 16), 0.5, dtype=np.float32)
        self._last_obs_result = None
        self._need_frame_after_t: float = 0.0
        self._force_vision_next: bool = False
        self._pre_gimbal_frame = None
        self._no_fresh_vision_steps = 0
        self._last_failsafe_warn_t = 0.0
        
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
        self._consecutive_backward = 0
        self._consecutive_rotations = 0
        self._prev_action = -1
        self._last_us_cm  = 400.0

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
            if self._us and self._motors:
                # Blind spot handling and escape logic
                _min_r = float(self._cfg.get("hal", {}).get(
                               "ultrasonic", {}).get("min_range_cm", 3.0))
                try:
                    for attempt in range(5):
                        d1 = self._us.read_cm()
                        time.sleep(0.05)
                        d2 = self._us.read_cm()
                        if d1 <= _min_r or d2 <= _min_r:
                            break
                        if d1 >= 25.0 or d2 >= 25.0:
                            if d1 < 25.0 or d2 < 25.0:
                                break
                        dist = max(d1, d2)
                        logger.info("[Env] Wall escape attempt %d: dist=%.1f cm", attempt + 1, dist)
                        self._motors.move_backward(40)
                        time.sleep(0.4)
                        self._motors.stop()
                        time.sleep(0.15)
                    
                    # Ensure safe start distance
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
            self._grid.soft_reset()

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply action, observe result.
        """
        # Execute action
        collided = self._execute_action(action)

        # Progress pose estimate
        self._update_pose(action)

        # Sense
        us_dists, obs_result = self._sense()

        # Map update
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

        # Stuck-backward detector
        if action == ACT_BACKWARD:
            us_now = us_dists[0] if us_dists else 400.0
            if abs(us_now - self._last_us_cm) < 3.0:
                self._consecutive_backward += 1
            else:
                self._consecutive_backward = 0
            self._last_us_cm = us_now
        else:
            self._consecutive_backward = 0

        self._step_count += 1
        
        emergency = nearest_cm < self._emergency_cm
        stuck_backward = (self._consecutive_backward >= 6 and self._mode == "robot")
        
        if stuck_backward:
            logger.warning("[Env] Stuck backing into wall after %d steps — forcing reset", self._consecutive_backward)
            
        done = (emergency or collided or stuck_backward or self._step_count >= self._max_steps)
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
        
        if self._cfg.get("render_sim", False) and self._mode == "sim": 
            self.render()
            
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
        if obs_result is not None and getattr(obs_result, 'visual_features', None) is not None:
            parts.append(np.array(obs_result.visual_features, dtype=np.float32)[:self._feat_dim])
        else:
            parts.append(np.zeros(self._feat_dim, dtype=np.float32))

        # Depth map (flat)
        dm_src = obs_result.depth_map if obs_result is not None else self._last_depth_map
        if dm_src is None:
             dm_src = np.full((16, 16), 0.5, dtype=np.float32)
             
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
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        state = np.clip(state, -5.0, 5.0)
        return state.astype(np.float32)

    # ── Action execution ──────────────────────────────────────────

    def _execute_action(self, action: int) -> bool:
        if self._mode == "robot":
            return self._execute_action_robot(action)
        return self._execute_action_sim(action)

    def _execute_action_robot(self, action: int) -> bool:
        # Failsafe logic omitted for brevity (same as original)
        if action in (ACT_FORWARD, ACT_BACKWARD, ACT_STRAFE_LEFT, ACT_STRAFE_RIGHT, ACT_ROTATE_LEFT, ACT_ROTATE_RIGHT):
            if self._need_frame_after_t > 0.0 or bool(getattr(self, '_force_vision_next', False)):
                 if self._motors: self._motors.stop()
                 now = time.monotonic()
                 if (now - self._last_failsafe_warn_t) > 1.0:
                     logger.warning('[FailSafe] Blocking motion until post-pan fresh vision is available')
                     self._last_failsafe_warn_t = now
                 return False

        if self._controller:
            self._controller.move_action(action, self._base_speed,
                                          self._rotate_speed,
                                          self._strafe_speed,
                                          self._reverse_speed)
        
        # Gimbal movement logic
        if self._gimbal:
            step = self._gimbal_step
            moved_gimbal = False
            if action == ACT_PAN_LEFT:
                self._gimbal.move_pan(-step); moved_gimbal = True
            elif action == ACT_PAN_RIGHT:
                self._gimbal.move_pan(step); moved_gimbal = True
            elif action == ACT_TILT_UP:
                self._gimbal.move_tilt(step); moved_gimbal = True
            elif action == ACT_TILT_DOWN:
                self._gimbal.move_tilt(-step); moved_gimbal = True
            
            if moved_gimbal:
                if self._camera:
                    try: self._pre_gimbal_frame = self._camera.read()
                    except Exception: self._pre_gimbal_frame = None
                self._need_frame_after_t = time.monotonic()
                self._force_vision_next = True

        try:
            time.sleep(0.08)
        finally:
            if self._motors and action != ACT_STOP:
                self._motors.stop()
        return False

    def _execute_action_sim(self, action: int) -> bool:
        """Simulate motion kinematics."""
        dt   = 0.1
        spd  = 0.3
        rot  = 0.8
        dx = dy = dtheta = 0.0
        if action == ACT_FORWARD:     dx = spd * dt
        elif action == ACT_BACKWARD:  dx = -spd * dt
        elif action == ACT_STRAFE_LEFT:  dy = -spd * dt
        elif action == ACT_STRAFE_RIGHT: dy =  spd * dt
        elif action == ACT_ROTATE_LEFT:  dtheta =  rot * dt
        elif action == ACT_ROTATE_RIGHT: dtheta = -rot * dt

        ct = math.cos(self._robot_theta)
        st = math.sin(self._robot_theta)
        wx = dx * ct - dy * st
        wy = dx * st + dy * ct
        nx = self._robot_x + wx
        ny = self._robot_y + wy
        ntheta = self._robot_theta + dtheta

        for ox, oy, radius in self._sim_obstacles:
            if math.hypot(nx - ox, ny - oy) < radius + 0.18:
                return True
        self._robot_x = nx
        self._robot_y = ny
        self._robot_theta = ntheta
        self._vx  = wx / dt
        self._vy  = wy / dt
        self._omega = dtheta / dt
        return False

    def _update_pose(self, action: int) -> None:
        pass

    # ── Sensing ───────────────────────────────────────────────────

    def _sense(self):
        """Collect sensor readings and run vision pipeline."""
        if self._mode == "robot" and self._us:
            us_dist = self._us.read_cm()
        elif self._mode == "sim":
            us_dist = self._sim_us_forward()
        else:
            us_dist = 200.0

        us_dists = [us_dist]
        obs_result = self._last_obs_result

        rl_cfg = self._cfg.get("rl", {})
        vision_interval = (
            rl_cfg.get("vision_every_n_steps")
            if rl_cfg.get("vision_every_n_steps") is not None
            else rl_cfg.get("ppo", {}).get("vision_every_n_steps")
        )
        try: vision_interval = int(vision_interval) if vision_interval is not None else 3
        except Exception: vision_interval = 3
        vision_interval = max(1, vision_interval)

        force_vision = bool(self._force_vision_next)
        self._force_vision_next = False
        should_run_vision = force_vision or ((self._step_count % vision_interval) == 0)

        # SIMULATION VISION (Patched for Speed & Accuracy)
        if self._mode == "sim" and should_run_vision:
            # We skip neural network vision entirely in sim for performance and ground-truth purity.
            frame, depth_map_16x16 = self._render_camera_sim()
            
            # Construct a synthetic PerceptionResult
            # If PerceptionResult class is unavailable (e.g. running on restricted env), use Namespace
            factory = PerceptionResult if PerceptionResult else SimpleNamespace
            
            # Create synthetic result
            # Depth: already 16x16 float 0-1
            # Features: Zeroes (network not run)
            # Obstacles: Derived from depth (simple threshold)
            
            obs_result = factory(
                timestamp=time.time(),
                depth_map=depth_map_16x16,
                obstacle_boxes=[], # TODO: Could generate boxes from sim obstacles
                floor_map=None,
                visual_features=np.zeros(self._feat_dim, dtype=np.float32),
                nearest_obstacle_cm=us_dist,
                floor_clear=True,
                processing_ms=0.0
            )
            self._last_obs_result = obs_result
            self._last_depth_map = depth_map_16x16

        # ROBOT VISION (Real)
        elif self._camera and self._vision and should_run_vision:
            frame = None
            if self._need_frame_after_t > 0.0:
                 after_t = float(self._need_frame_after_t)
                 frame = self._camera.read_fresh(after_t=after_t, timeout_s=0.6)
                 if frame is None:
                     self._force_vision_next = True
                     self._no_fresh_vision_steps += 1
                     if self._no_fresh_vision_steps >= 20:
                         if self._motors: self._motors.stop()
                         logger.error('[FailSafe] Vision stale > 20 steps')
                 else:
                     self._need_frame_after_t = 0.0
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
        rng = np.random.default_rng(0)
        n = 12
        xs = rng.uniform(-4, 4, n)
        ys = rng.uniform(-4, 4, n)
        rs = rng.uniform(0.15, 0.40, n)
        self._sim_obstacles = list(zip(xs.tolist(), ys.tolist(), rs.tolist()))
        # Hard walls at ±5 m
        wall_pts = [(-5, 0, 0.1), (5, 0, 0.1), (0, -5, 0.1), (0, 5, 0.1)]
        self._sim_obstacles.extend(wall_pts)

    def _sim_place_robot(self) -> None:
        rng = np.random.default_rng(int(time.time() * 1000) % 2**31)
        for _ in range(100):
            x = rng.uniform(-3, 3)
            y = rng.uniform(-3, 3)
            if all(math.hypot(x-ox, y-oy) > 0.4 for ox, oy, _ in self._sim_obstacles):
                self._robot_x = x; self._robot_y = y
                self._robot_theta = rng.uniform(-math.pi, math.pi)
                return
        self._robot_x = 0.0; self._robot_y = 0.0; self._robot_theta = 0.0

    def _sim_us_forward(self) -> float:
        angle_world = self._robot_theta
        min_dist = 400.0
        for ox, oy, radius in self._sim_obstacles:
            dx = ox - self._robot_x
            dy = oy - self._robot_y
            proj = dx * math.cos(angle_world) + dy * math.sin(angle_world)
            if proj <= 0: continue
            perp2 = (dx**2 + dy**2) - proj**2
            if perp2 > radius**2: continue
            hit = proj - math.sqrt(max(0, radius**2 - perp2))
            dist_cm = hit * 100.0
            if 2 <= dist_cm < min_dist:
                min_dist = dist_cm
        return min_dist

    def render(self) -> None:
        try: import cv2
        except ImportError: return

        width = 500
        mid = width // 2
        scale = 50.0 # 50px/m
        img = np.full((width, width, 3), 255, dtype=np.uint8)

        for ox, oy, r in self._sim_obstacles:
            cx = int(mid + ox * scale)
            cy = int(mid - oy * scale)
            cr = int(r * scale)
            cv2.circle(img, (cx, cy), cr, (100, 100, 100), -1)

        rx = int(mid + self._robot_x * scale)
        ry = int(mid - self._robot_y * scale)
        cv2.circle(img, (rx, ry), 8, (255, 0, 0), -1) # Robot
        
        dist_m = self._sim_us_forward() / 100.0
        draw_dist = min(dist_m, 2.0)
        ex = int(rx + (draw_dist * scale) * math.cos(self._robot_theta))
        ey = int(ry - (draw_dist * scale) * math.sin(self._robot_theta))
        cv2.line(img, (rx, ry), (ex, ey), (0, 255, 0), 1)
        
        cv2.imshow("Robot Sim (10x10m Room)", img)
        cv2.waitKey(1)

    def _render_camera_sim(self) -> Tuple[Any, np.ndarray]:
        """
        Render a simulated first-person view AND a synthetic depth map.
        Returns:
            (image_bgr, depth_map_16x16_float_0to1)
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None, np.full((16, 16), 0.5, dtype=np.float32)

        width, height = 320, 240
        fov_deg = 60.0
        max_dist = 5.0
        
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (0, height//2), (width, height), (240, 240, 240), -1) # Floor

        # ─── 1. Render Visual Image (Raycast columns) ───
        n_rays_viz = width // 4
        angle_step_viz = math.radians(fov_deg) / n_rays_viz
        start_angle_viz = self._robot_theta + math.radians(fov_deg/2)
        
        # We also want to compute the 16x16 depth map directly via raycasting
        # instead of downsampling the image (which is messy).
        # We need 16 columns of depth. Let's cast 16 rays evenly spaced in FOV.
        
        depth_map_line = np.zeros(16, dtype=np.float32)
        
        # Ray casting for the visual image
        for i in range(n_rays_viz):
            angle = start_angle_viz - i * angle_step_viz
            min_dist = max_dist
            
            for ox, oy, r in self._sim_obstacles:
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
                h = int(height / (min_dist + 0.1))
                h = min(h, height)
                color = int(255 * (min_dist / max_dist))
                color = max(0, min(255, color))
                x_start = i * 4
                x_end = (i+1) * 4
                cv2.rectangle(img, (x_start, height//2 - h//2), 
                                   (x_end, height//2 + h//2), 
                                   (color, color, color), -1)

        # ─── 2. Compute Synthetic Depth Map (16x16) ───
        # We need a 16x16 grid.
        n_rays_depth = 16
        angle_step_depth = math.radians(fov_deg) / n_rays_depth
        start_angle_depth = self._robot_theta + math.radians(fov_deg/2) 

        for c in range(n_rays_depth):
            # Center of the bin
            angle = start_angle_depth - (c + 0.5) * angle_step_depth
            min_dist = max_dist
            
            for ox, oy, r in self._sim_obstacles:
                dx = ox - self._robot_x
                dy = oy - self._robot_y
                proj = dx * math.cos(angle) + dy * math.sin(angle)
                if proj <= 0: continue
                perp2 = (dx**2 + dy**2) - proj**2
                if perp2 > r**2: continue
                hit = proj - math.sqrt(max(0, r**2 - perp2))
                if hit < min_dist:
                    min_dist = hit
            
            # Map distance to 0-1 (Inverse depth style: 1=Close, 0=Far)
            inv_depth = 1.0 - (min_dist / max_dist)
            inv_depth = max(0.0, min(1.0, inv_depth))
            depth_map_line[c] = inv_depth

        # Create 16x16 by stacking. 
        depth_map = np.tile(depth_map_line, (16, 1)).astype(np.float32)
        
        return img, depth_map
