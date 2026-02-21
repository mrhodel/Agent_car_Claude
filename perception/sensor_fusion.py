"""
perception/sensor_fusion.py
Fuses ultrasonic distance readings with camera-derived obstacle estimates
to produce a unified, temporally-smoothed obstacle map.

Hardware context
----------------
The Yahboom Raspbot V2 has a SINGLE, FIXED ultrasonic sensor that always
points forward (0° relative to the robot heading).  It does NOT sweep with
the gimbal.  Therefore, the ultrasonic contributes exactly ONE reading per
tick at the robot's current heading direction.

The fusion strategy:
  • Ultrasonic: high-confidence single forward reading at ``robot_heading_deg``.
  • Camera obstacles: assigned confidence based on detection score.
  • A sliding window keeps the last N observations; the fused estimate
    is the weighted mean inside each angular bin.

Output: FusedObstacleMap – per-angle distance array covering the
        robot's horizontal FOV (populated by camera; US = 1 forward point).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FusedObstacleMap:
    """Polar obstacle map centered on the robot."""
    # Each entry: (angle_deg, distance_cm) – sorted by angle
    angles: np.ndarray          # shape (N,)  degrees
    distances: np.ndarray       # shape (N,)  cm
    confidence: np.ndarray      # shape (N,)  0–1

    def nearest(self) -> float:
        """Return the distance to the closest obstacle across all angles."""
        if len(self.distances) == 0:
            return float("inf")
        return float(self.distances.min())

    def distance_at_angle(self, angle_deg: float, tol_deg: float = 15.0) -> float:
        """Return fused distance reading nearest to ``angle_deg``."""
        diffs = np.abs(self.angles - angle_deg)
        idx = int(diffs.argmin())
        if diffs[idx] <= tol_deg:
            return float(self.distances[idx])
        return float("inf")

    def to_vector(self) -> np.ndarray:
        """Flatten to a 1-D feature vector suitable for RL state."""
        return np.stack([self.angles / 90.0,
                          np.clip(self.distances / 400.0, 0, 1),
                          self.confidence], axis=1).flatten()


class SensorFusion:
    """
    Parameters
    ----------
    cfg : dict
        The ``perception.sensor_fusion`` sub-dict from robot_config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._us_weight   = float(cfg.get("us_weight",    0.60))
        self._cam_weight  = float(cfg.get("cam_weight",   0.40))
        self._window      = int(cfg.get("history_window", 5))
        # angle_bin → deque of (dist, confidence) observations
        self._history: Dict[int, deque] = {}

    # ── Public API ────────────────────────────────────────────────

    def update(
        self,
        us_distance: float,
        robot_heading_deg: float,
        cam_angles: List[float],
        cam_distances: List[float],
        cam_confidences: List[float],
    ) -> FusedObstacleMap:
        """
        Incorporate new readings and return the updated fused map.

        Parameters
        ----------
        us_distance       : ultrasonic forward distance (cm) – single fixed sensor
        robot_heading_deg : robot heading in world frame (degrees), used as the
                            angular bin for the US reading
        cam_angles        : obstacle horizontal angles from camera (degrees, world frame)
        cam_distances     : camera-estimated distances (cm)
        cam_confidences   : detection confidence for each camera obstacle
        """
        # --- Ultrasonic: single forward reading at heading angle ---
        bin_key = self._angle_bin(robot_heading_deg)
        if bin_key not in self._history:
            self._history[bin_key] = deque(maxlen=self._window)
        self._history[bin_key].append((us_distance, self._us_weight))

        # --- Accumulate camera observations ---
        for angle, dist, conf in zip(cam_angles, cam_distances, cam_confidences):
            bin_key = self._angle_bin(angle)
            if bin_key not in self._history:
                self._history[bin_key] = deque(maxlen=self._window)
            self._history[bin_key].append((dist, conf * self._cam_weight))

        # --- Compute fused output ---
        angles_out     = []
        distances_out  = []
        confidence_out = []

        for bin_key in sorted(self._history.keys()):
            obs = list(self._history[bin_key])
            if not obs:
                continue
            dists, weights = zip(*obs)
            w_arr = np.array(weights, dtype=np.float32)
            d_arr = np.array(dists,   dtype=np.float32)
            # Weighted mean
            w_sum = w_arr.sum()
            fused_dist = float((d_arr * w_arr).sum() / w_sum) if w_sum > 0 else float("inf")
            fused_conf = float(w_arr.max())
            angles_out.append(float(bin_key))
            distances_out.append(fused_dist)
            confidence_out.append(fused_conf)

        if not angles_out:
            # No data yet – return an empty map
            return FusedObstacleMap(
                angles=np.array([], dtype=np.float32),
                distances=np.array([], dtype=np.float32),
                confidence=np.array([], dtype=np.float32),
            )

        return FusedObstacleMap(
            angles=np.array(angles_out,     dtype=np.float32),
            distances=np.array(distances_out, dtype=np.float32),
            confidence=np.array(confidence_out, dtype=np.float32),
        )

    def reset(self) -> None:
        """Clear all history (call at episode start for training)."""
        self._history.clear()

    # ── Internal ─────────────────────────────────────────────────

    @staticmethod
    def _angle_bin(angle_deg: float, bin_size: float = 10.0) -> int:
        """Quantise a continuous angle to the nearest bin centre (degrees)."""
        return int(round(angle_deg / bin_size) * bin_size)
