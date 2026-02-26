"""
rl/reward.py
Reward shaping for autonomous indoor navigation.

Design goals:
  +  Encourage exploration (new cells discovered)
  +  Reward goal reaching
  -  Penalise collisions heavily
  -  Penalise lingering near obstacles
  +  Small momentum bonus for repeating same locomotion action
  -  Small backward cost — discourages wall-backing exploit without punishing genuine retreats too harshly
  -  Penalise sustained rotation (discourages spinning in place)
  -  Extra cost for gimbal-only actions (discourage using them as cheap filler)
  -  Small per-step time cost (encourages efficiency)
"""
from __future__ import annotations

import math

# Action indices
ACT_FORWARD      = 0
ACT_BACKWARD     = 1
ACT_STRAFE_LEFT  = 2
ACT_STRAFE_RIGHT = 3
ACT_ROTATE_LEFT  = 4
ACT_ROTATE_RIGHT = 5
_ROTATE_ACTIONS  = {ACT_ROTATE_LEFT, ACT_ROTATE_RIGHT}
# Only pure rotation counts as "spinning" — strafing is a useful mecanum move
_SPINNING_ACTIONS = {ACT_ROTATE_LEFT, ACT_ROTATE_RIGHT}
# Gimbal pan/tilt actions — useful for perception but must not replace movement
_GIMBAL_ACTIONS   = {6, 7, 8, 9}   # pan_L, pan_R, tilt_up, tilt_down


class RewardCalculator:

    @staticmethod
    def compute(
        new_explored_cells: int,
        nearest_obstacle_cm: float,
        collided: bool,
        action: int,
        prev_action: int,
        cfg: dict,
        consecutive_rotations: int = 0,
    ) -> float:
        r = 0.0

        # ── Exploration bonus ────────────────────────────────────
        exp_bonus = float(cfg.get("exploration_bonus", 1.0))
        r += new_explored_cells * exp_bonus

        # ── Collision penalty ────────────────────────────────────
        if collided:
            r += float(cfg.get("collision_penalty", -15.0))

        # ── Proximity penalty ────────────────────────────────────
        prox_scale = float(cfg.get("proximity_penalty_scale", -0.5))
        danger_cm  = 30.0
        if nearest_obstacle_cm < danger_cm:
            # Gradient: stronger penalty as obstacle gets closer
            r += prox_scale * (1.0 / max(1.0, nearest_obstacle_cm))

        # ── Time cost ────────────────────────────────────────────
        r += float(cfg.get("time_step_cost", -0.01))

        # ── Momentum bonus (same locomotion action repeated) ─────
        # Small bonus for continuing the same direction of travel.
        # Deliberately NO reversal penalty: backing away from a wall and
        # then going forward is exactly the desired recovery behaviour.
        # Gimbal actions are excluded — no momentum reward for camera moves.
        smooth_bonus = float(cfg.get("smooth_motion_bonus", 0.05))
        if action not in _SPINNING_ACTIONS and action not in _GIMBAL_ACTIONS:
            if action == prev_action:
                r += smooth_bonus * 0.5

        # ── Forward bonus ──────────────────────────────────────────
        # Small reward for choosing to move forward, nudging the policy
        # toward exploration over passive avoidance.
        if action == ACT_FORWARD:
            r += float(cfg.get("forward_bonus", 0.02))

        # ── Backward-step cost ───────────────────────────────────────
        # Recurring penalty for reversing — makes wall-backing unprofitable
        # without blocking short retreats for genuine obstacle avoidance.
        if action == ACT_BACKWARD:
            r += float(cfg.get("backward_step_cost", -0.05))

        # ── Gimbal-only action extra cost ─────────────────────────
        # Camera pan/tilt gives the policy perceptual info but the policy
        # was using gimbal moves as cheap "safe" filler (~40/143 steps).
        # Extra cost makes locomotion relatively more attractive.
        if action in _GIMBAL_ACTIONS:
            r += float(cfg.get("gimbal_step_cost", -0.04))

        # ── Sustained rotation penalty ────────────────────────────
        # Small per-step penalty that grows with consecutive rotations,
        # discouraging the policy from spinning in place.
        if action in _SPINNING_ACTIONS and consecutive_rotations > 2:
            spin_penalty = float(cfg.get("spin_penalty", -0.1))
            r += spin_penalty * min(consecutive_rotations - 2, 8)

        return r


def _is_reversal(a: int, b: int) -> bool:
    """Return True if (a, b) are opposite actions (e.g. forward/backward)."""
    pairs = [(0, 1), (2, 3), (4, 5)]   # (forward, backward), etc.
    return (a, b) in pairs or (b, a) in pairs
