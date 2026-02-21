"""
rl/reward.py
Reward shaping for autonomous indoor navigation.

Design goals:
  +  Encourage exploration (new cells discovered)
  +  Reward goal reaching
  -  Penalise collisions heavily
  -  Penalise lingering near obstacles
  +  Small bonus for smooth motion (avoids jittery behaviour)
  -  Small per-step time cost (encourages efficiency)
"""
from __future__ import annotations

import math


class RewardCalculator:

    @staticmethod
    def compute(
        new_explored_cells: int,
        nearest_obstacle_cm: float,
        collided: bool,
        action: int,
        prev_action: int,
        cfg: dict,
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

        # ── Smooth motion bonus ──────────────────────────────────
        # Bonus if action continues same direction; penalty for reversal
        smooth_bonus = float(cfg.get("smooth_motion_bonus", 0.05))
        if action == prev_action:
            r += smooth_bonus * 0.5
        elif _is_reversal(action, prev_action):
            r -= smooth_bonus

        return r


def _is_reversal(a: int, b: int) -> bool:
    """Return True if (a, b) are opposite actions (e.g. forward/backward)."""
    pairs = [(0, 1), (2, 3), (4, 5)]   # (forward, backward), etc.
    return (a, b) in pairs or (b, a) in pairs
