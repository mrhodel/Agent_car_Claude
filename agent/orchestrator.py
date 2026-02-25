"""
agent/orchestrator.py
Top-level Agent Orchestrator for the Yahboom Raspbot V2.

Finite State Machine:
  ┌─────────┐       ┌──────────┐       ┌──────────┐
  │  INIT   │──────▶│ EXPLORE  │──────▶│  PLAN    │
  └─────────┘       └──────────┘       └──────────┘
                         ▲                  │
                         │                  ▼
                    ┌──────────┐       ┌──────────┐
                    │ RECOVER  │◀──────│ EXECUTE  │
                    └──────────┘       └──────────┘

INIT     : hardware self-test, gimbal sweep, initial map seed
EXPLORE  : RL policy selects actions; map is updated each step
PLAN     : A* toward chosen frontier when RL suggests replanning
EXECUTE  : pure-pursuit follows the A* path
RECOVER  : spin-and-backup when stuck or near collision

The orchestrator runs at the configured loop frequency (default 10 Hz)
and handles:
  • Emergency stop on proximity < 10 cm
  • Periodic full gimbal scan for mapping
  • Checkpoint loading/saving
  • Clean shutdown on SIGINT/SIGTERM
"""
from __future__ import annotations

import logging
import math
import os
import signal
import time
from enum import Enum, auto
from typing import Optional

import numpy as np

from hal          import MotorController, UltrasonicSensor, Gimbal, Camera
from perception   import VisionPipeline, SensorFusion
from mapping      import OccupancyGrid
from mapping.occupancy_grid import RobotPose
from navigation   import AStarPlanner, MotionController
from rl           import RobotEnv, ActorCritic, PPOAgent

logger = logging.getLogger(__name__)


class AgentState(Enum):
    INIT    = auto()
    EXPLORE = auto()
    PLAN    = auto()
    EXECUTE = auto()
    RECOVER = auto()
    STOPPED = auto()


class AgentOrchestrator:
    """
    Main agent controller.

    Parameters
    ----------
    cfg : dict
        Full parsed robot_config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg   = cfg
        self._state = AgentState.INIT
        self._running = False

        agent_cfg = cfg.get("agent", {})
        self._hz         = float(agent_cfg.get("loop_hz", 10))
        self._dt         = 1.0 / self._hz
        self._emergency  = float(agent_cfg.get("safety", {})
                                  .get("emergency_stop_cm", 10))
        self._exp_cfg    = agent_cfg.get("exploration", {})
        self._log_cfg    = agent_cfg.get("logging", {})

        # ── Instantiate hardware ─────────────────────────────────
        hal_cfg = cfg.get("hal", {})
        self._motors = MotorController(hal_cfg.get("motor", {}))
        self._us     = UltrasonicSensor(hal_cfg.get("ultrasonic", {}))
        self._gimbal = Gimbal(hal_cfg.get("gimbal", {}))
        self._camera = Camera(hal_cfg.get("camera", {}))

        # ── Perception ───────────────────────────────────────────
        self._vision  = VisionPipeline(cfg.get("perception", {}))
        # Ultrasonic is a fixed forward sensor – no ray_angles list needed
        self._fusion = SensorFusion(
            cfg.get("perception", {}).get("sensor_fusion", {}))

        # ── Mapping ──────────────────────────────────────────────
        self._grid = OccupancyGrid(cfg.get("mapping", {}).get("grid", {}))

        # ── Navigation ───────────────────────────────────────────
        nav_cfg = cfg.get("navigation", {})
        self._planner    = AStarPlanner(nav_cfg.get("planner", {}))
        self._controller = MotionController(
            self._motors, nav_cfg.get("controller", {}))
        self._controller.set_recovery_config(nav_cfg.get("recovery", {}))

        # ── RL ───────────────────────────────────────────────────
        rl_cfg = cfg.get("rl", {})
        state_cfg = rl_cfg.get("state", {})
        n_rays     = int(state_cfg.get("ultrasonic_rays",    1))  # fixed sensor = 1
        feat_dim   = int(state_cfg.get("visual_feature_dim", 128))
        depth_flat = int(state_cfg.get("depth_map_flat",     256))
        map_sz     = int(state_cfg.get("local_map_size",     7))
        extra = 0
        if state_cfg.get("include_velocity", True): extra += 3
        if state_cfg.get("include_heading",  True): extra += 2
        state_dim = n_rays + feat_dim + depth_flat + map_sz**2 + extra

        self._env = RobotEnv(
            cfg=cfg,
            motors=self._motors,
            ultrasonic=self._us,
            gimbal=self._gimbal,
            camera=self._camera,
            vision=self._vision,
            grid=self._grid,
            controller=self._controller,
            mode="robot",
        )
        try:
            from rl.policy import ActorCritic
            self._policy = ActorCritic(
                state_dim=state_dim,
                n_actions=self._env.n_actions,
                hidden_dim=int(rl_cfg.get("ppo", {}).get("hidden_dim", 256)),
            )
        except Exception:
            from rl.policy import ActorCriticNumpy
            self._policy = ActorCriticNumpy(state_dim, self._env.n_actions)

        self._agent = PPOAgent(
            policy=self._policy,
            cfg=rl_cfg.get("ppo", {}),
            state_dim=state_dim,
            n_actions=self._env.n_actions,
        )

        # State
        self._pose        = RobotPose()
        self._current_path = None
        self._last_scan_t  = 0.0
        self._rescan_iv    = float(self._exp_cfg.get("rescan_interval_s", 5.0))
        self._save_map_t   = 0.0
        self._save_map_iv  = float(self._log_cfg.get("save_map_interval_s", 30))
        self._ep_reward    = 0.0
        self._ep_steps     = 0
        self._rl_state: Optional[np.ndarray] = None
        self._spin_count   = 0  # consecutive spinning actions for run-mode guard

        # Register shutdown hooks
        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        logger.info("[Orchestrator] Initialised  state_dim=%d", state_dim)

    # ── Public API ────────────────────────────────────────────────

    def run(self, checkpoint: Optional[str] = None) -> None:
        """
        Start the autonomous navigation loop.

        Parameters
        ----------
        checkpoint : optional path to a .pt / .npz weights file to load
                     before starting.
        """
        if checkpoint and os.path.exists(checkpoint):
            self._agent.load_checkpoint(checkpoint)

        self._running = True
        self._state   = AgentState.INIT
        logger.info("[Orchestrator] Running at %.1f Hz", self._hz)

        while self._running:
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as exc:
                logger.exception("[Orchestrator] Unhandled error in tick: %s", exc)
                self._motors.stop()
            elapsed = time.monotonic() - t0
            sleep_t = self._dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        self._shutdown()

    def stop(self) -> None:
        self._running = False

    # ── FSM tick ─────────────────────────────────────────────────

    def _tick(self) -> None:
        # --- Always: read sensors and check safety ---
        # Ultrasonic is a FIXED forward sensor – single reading, no sweep
        us_dist  = self._us.read_cm()
        us_dists = [us_dist]   # 1-element list for downstream consumers
        nearest  = us_dist

        # Emergency stop – skip if already recovering (let recovery handler run)
        if nearest < self._emergency and self._state != AgentState.RECOVER:
            logger.warning("[Safety] Emergency stop  nearest=%.1f cm", nearest)
            self._motors.stop()
            self._transition(AgentState.RECOVER)
            return

        # Read camera and run vision
        frame = self._camera.read()
        obs_result = None
        if frame is not None:
            obs_result = self._vision.process(frame, us_dist)

        # Sensor fusion – single US forward reading at robot heading
        heading_deg = math.degrees(self._pose.theta)
        fused = self._fusion.update(
            us_distance=us_dist,
            robot_heading_deg=heading_deg,
            cam_angles=[b.angle_deg for b in (obs_result.obstacle_boxes
                                               if obs_result else [])],
            cam_distances=[b.dist_cm for b in (obs_result.obstacle_boxes
                                                if obs_result else [])],
            cam_confidences=[b.confidence for b in (obs_result.obstacle_boxes
                                                      if obs_result else [])],
        )

        # Map update – US contributes a single forward ray
        self._grid.update(
            self._pose,
            obstacle_angles_deg=[heading_deg] if us_dist < 300 else [],
            obstacle_dists_cm=[us_dist]       if us_dist < 300 else [],
        )

        # Periodic gimbal scan (camera-only sweep; US is fixed)
        if time.monotonic() - self._last_scan_t > self._rescan_iv:
            logger.info("[Scan] Periodic gimbal sweep (camera coverage, US is fixed forward)")
            self._full_scan()

        # Periodic map save
        if time.monotonic() - self._save_map_t > self._save_map_iv:
            self._save_map()
            self._save_map_t = time.monotonic()

        # --- FSM transitions ---
        if self._state == AgentState.INIT:
            self._handle_init()
        elif self._state == AgentState.EXPLORE:
            self._handle_explore(us_dists, obs_result)
        elif self._state == AgentState.PLAN:
            self._handle_plan()
        elif self._state == AgentState.EXECUTE:
            self._handle_execute(nearest)
        elif self._state == AgentState.RECOVER:
            self._handle_recover()

    # ── FSM handlers ─────────────────────────────────────────────

    def _handle_init(self) -> None:
        logger.info("[FSM] INIT -> performing camera gimbal sweep + map seed")
        self._full_scan()
        self._rl_state = self._env.reset()
        self._transition(AgentState.EXPLORE)

    # Human-readable action names for diagnostics
    _ACTION_NAMES = {
        0: "forward",
        1: "backward",
        2: "strafe_left",
        3: "strafe_right",
        4: "rotate_left",
        5: "rotate_right",
        6: "gimbal_pan_left",
        7: "gimbal_pan_right",
        8: "gimbal_tilt_up",
        9: "gimbal_tilt_down",
        10: "stop",
    }

    def _handle_explore(self, us_dists, obs_result) -> None:
        """Use RL policy to pick the next action."""
        if self._rl_state is None:
            self._rl_state = self._env.reset()

        action, log_prob, value = self._agent.select_action(self._rl_state)
        nearest_cm = us_dists[0] if us_dists else 999.0

        # Anti-spin guard: if 4+ consecutive rotate actions, force forward
        _SPIN_ACTIONS = {4, 5}  # rotate_left, rotate_right only (strafe is useful)
        overridden = False
        if action in _SPIN_ACTIONS:
            self._spin_count += 1
        else:
            self._spin_count = 0
        if self._spin_count >= 4:
            action = 0
            self._spin_count = 0
            overridden = True

        action_name = self._ACTION_NAMES.get(action, str(action))
        logger.info(
            "[Explore] step=%-4d  action=%-14s  us=%-6.1f cm  value=%+.3f  logp=%+.3f%s",
            self._ep_steps, action_name, nearest_cm, value, log_prob,
            "  (anti-spin override)" if overridden else "",
        )

        next_state, reward, done, info = self._env.step(action)

        logger.info(
            "[Explore] step=%-4d  reward=%+7.3f  ep_total=%+8.3f  done=%s",
            self._ep_steps, reward, self._ep_reward + reward, done,
        )

        self._agent.store(self._rl_state, action, log_prob, reward, done, value)
        self._ep_reward += reward
        self._ep_steps  += 1
        self._rl_state   = next_state

        # Trigger PPO update if buffer is full
        if self._agent.ready_to_update():
            _, _, last_val = self._agent.select_action(next_state)
            self._agent.update(last_val if not done else 0.0)
            logger.info("[Explore] PPO update triggered at step %d", self._ep_steps)

        # Check if we should switch to planned navigation
        frontiers = self._grid.get_frontiers(
            self._pose,
            min_size=int(self._exp_cfg.get("frontier_min_size", 3))
        )
        if frontiers and self._ep_steps % 50 == 0:
            self._target_frontier = frontiers[0]
            logger.info(
                "[Explore] Frontier selected  grid=(%d,%d)  total_frontiers=%d",
                self._target_frontier.row, self._target_frontier.col, len(frontiers),
            )
            self._transition(AgentState.PLAN)

        if done:
            logger.info(
                "[Explore] Episode done  steps=%d  total_reward=%.2f  explored_cells=%d",
                self._ep_steps, self._ep_reward,
                self._grid.explored_cells if self._grid else 0,
            )
            self._agent.on_episode_end(self._ep_reward, self._ep_steps)
            self._ep_reward = 0.0
            self._ep_steps  = 0
            self._rl_state  = self._env.reset()

    def _handle_plan(self) -> None:
        """Run A* toward the chosen frontier."""
        if not hasattr(self, "_target_frontier"):
            self._transition(AgentState.EXPLORE)
            return

        goal_world = self._grid.grid_to_world(
            self._target_frontier.row, self._target_frontier.col)
        path = self._planner.plan(self._grid, self._pose, goal_world)

        if path is None or path.empty:
            logger.debug("[FSM] No A* path found – back to EXPLORE")
            self._transition(AgentState.EXPLORE)
            return

        self._current_path = path
        self._controller.reset_recovery()
        logger.info("[FSM] A* path planned  waypoints=%d  len=%.1f m",
                     path.n_cells, path.length_m)
        self._transition(AgentState.EXECUTE)

    def _handle_execute(self, nearest_cm: float) -> None:
        """Follow the current A* path."""
        if self._current_path is None or self._current_path.empty:
            self._transition(AgentState.EXPLORE)
            return

        goal_reached = self._controller.follow_step(
            self._current_path,
            self._pose.x,
            self._pose.y,
            self._pose.theta,
            nearest_cm,
        )
        waypoints_left = len(self._current_path.waypoints) if self._current_path else 0
        logger.debug(
            "[Execute] us=%.1f cm  waypoints_remaining=%d  goal_reached=%s",
            nearest_cm, waypoints_left, goal_reached,
        )
        if goal_reached:
            logger.info("[FSM] Path goal reached – back to EXPLORE")
            self._current_path = None
            self._transition(AgentState.EXPLORE)
        elif nearest_cm < 20:
            logger.warning("[FSM] Obstacle during execution – RECOVER  nearest=%.1f cm", nearest_cm)
            self._motors.stop()
            self._transition(AgentState.RECOVER)

    def _handle_recover(self) -> None:
        us_dist = self._us.read_cm()
        logger.info("[Recover] Attempting recovery  us=%.1f cm  attempt=%d/%d",
                    us_dist, self._controller._recovery_attempts + 1,
                    self._controller._max_recovery)
        exhausted = self._controller.recover()
        if exhausted:
            logger.warning("[FSM] Recovery exhausted – resetting via env and back to EXPLORE")
            self._rl_state = self._env.reset()   # env reset backs robot away from wall
            self._controller.reset_recovery()    # clear counter so next encounter starts fresh
        else:
            self._controller.reset_recovery()    # clear after each successful recovery too
        self._current_path = None
        self._transition(AgentState.EXPLORE)

    # ── Helpers ───────────────────────────────────────────────────

    def _full_scan(self) -> None:
        """
        Sweep the gimbal pan axis so the camera covers a wide arc.
        This is a CAMERA-only sweep; the ultrasonic sensor does NOT move
        (it is fixed on the chassis, always pointing forward).

        The occupancy map is updated from the US forward reading taken
        once at the start, plus camera-derived obstacles detected while
        the gimbal sweeps.
        """
        # One US reading before we move the camera
        us_dist = self._us.read_cm()
        heading_deg = math.degrees(self._pose.theta)
        if us_dist < 300:
            self._grid.update(
                self._pose,
                obstacle_angles_deg=[heading_deg],
                obstacle_dists_cm=[us_dist],
            )

        # Sweep the camera gimbal for visual coverage
        n_steps  = int(self._exp_cfg.get("gimbal_scan_steps", 9))
        pan_min  = self._gimbal._pan_range[0]
        pan_max  = self._gimbal._pan_range[1]
        positions = np.linspace(pan_min, pan_max, n_steps).tolist()
        self._gimbal.sweep_pan(positions, dwell_s=0.06)
        self._gimbal.centre()
        self._last_scan_t = time.monotonic()

    def _save_map(self) -> None:
        log_dir = self._log_cfg.get("log_dir", "logs/")
        os.makedirs(log_dir, exist_ok=True)
        import cv2
        img = self._grid.to_image(self._pose)
        path = os.path.join(log_dir, "map_latest.png")
        cv2.imwrite(path, img)
        logger.debug("[Orchestrator] Map saved -> %s", path)

    def _transition(self, new_state: AgentState) -> None:
        if new_state != self._state:
            logger.info("[FSM] %s -> %s", self._state.name, new_state.name)
            self._state = new_state

    # ── Shutdown ─────────────────────────────────────────────────

    def _shutdown_handler(self, signum, frame) -> None:
        logger.info("[Orchestrator] Signal %d received – shutting down", signum)
        self._running = False

    def _shutdown(self) -> None:
        logger.info("[Orchestrator] Shutting down ...")
        self._motors.stop()
        self._gimbal.centre()
        self._camera.close()
        self._motors.close()
        self._gimbal.close()
        self._us.close()
        self._save_map()
        self._agent.save_checkpoint("final")
        logger.info("[Orchestrator] Clean shutdown complete")
