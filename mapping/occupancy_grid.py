"""
mapping/occupancy_grid.py
Probabilistic 2-D occupancy grid for indoor navigation.

The grid grows dynamically as the robot explores.  Cell values are
stored as log-odds to allow efficient Bayesian updates.  The robot
pose is tracked in continuous world coordinates; the grid expands
automatically when the robot approaches an edge.

Key concepts:
  • World frame    : origin at first robot position (metres).
  • Grid frame     : integer (row, col) indices.
  • Log-odds update: l += log(p_hit / (1 - p_hit))

Frontiers (boundary between known free and unknown cells) are exposed
for frontier-based exploration planning.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# Log-odds for Bayesian update
def _log_odds(p: float) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class GridCell:
    row: int
    col: int


@dataclass
class RobotPose:
    x: float = 0.0       # metres, world frame
    y: float = 0.0
    theta: float = 0.0   # radians, CCW from x-axis


class OccupancyGrid:
    """
    Dynamic probabilistic occupancy grid.

    Parameters
    ----------
    cfg : dict
        The ``mapping.grid`` sub-dict from robot_config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._res   = float(cfg.get("resolution_m", 0.05))   # m / cell
        self._size  = int(cfg.get("initial_size_cells", 400))
        self._exp_thresh  = int(cfg.get("expand_threshold",  20))
        self._exp_amount  = int(cfg.get("expand_amount",    100))
        self._decay_rate  = float(cfg.get("decay_rate", 0.001))
        self._blur_sigma  = float(cfg.get("blur_sigma", 0.8))
        self._obs_thresh  = float(cfg.get("obstacle_threshold", 0.65))
        self._free_thresh = float(cfg.get("free_threshold",     0.35))

        # Log-odds parameters
        self._lo_prior  = _log_odds(float(cfg.get("p_prior",    0.50)))
        self._lo_hit    = _log_odds(float(cfg.get("p_occ_hit",  0.75)))
        self._lo_miss   = _log_odds(float(cfg.get("p_occ_miss", 0.40)))
        self._lo_min    = _log_odds(float(cfg.get("p_occ_min",  0.05)))
        self._lo_max    = _log_odds(float(cfg.get("p_occ_max",  0.95)))

        # Grid storage (log-odds, float32)
        self._grid = np.full((self._size, self._size),
                              self._lo_prior, dtype=np.float32)
        # Known mask: True once a cell has been updated at least once
        self._known = np.zeros((self._size, self._size), dtype=bool)

        # Robot grid origin offset (world origin = centre of initial grid)
        self._origin_row = self._size // 2
        self._origin_col = self._size // 2

        # Frontier cache
        self._frontiers: List[GridCell] = []
        self._last_frontier_update = 0.0

        # Exploration coverage counter
        self._explored_cells = 0

        logger.info("[Map] %dx%d cells, %.2f m/cell → %.1f m × %.1f m",
                    self._size, self._size, self._res,
                    self._size * self._res, self._size * self._res)

    # ── Public API ────────────────────────────────────────────────

    def update(
        self,
        pose: RobotPose,
        obstacle_angles_deg: List[float],
        obstacle_dists_cm: List[float],
        free_angles_deg: Optional[List[float]] = None,
        free_dists_cm: Optional[List[float]] = None,
    ) -> int:
        """
        Update the grid from a set of obstacle and free-space ray readings.

        Parameters
        ----------
        pose               : current robot world pose
        obstacle_angles_deg: angles (world frame degrees) of detected obstacles
        obstacle_dists_cm  : corresponding distances (cm)
        free_angles_deg    : angles where no obstacle was detected
        free_dists_cm      : max range for each free ray

        Returns
        -------
        Number of newly explored cells (for RL exploration reward).
        """
        new_cells = 0

        # --- Mark obstacle cells ---
        for angle_deg, dist_cm in zip(obstacle_angles_deg, obstacle_dists_cm):
            cells = self._ray_cells(pose, angle_deg, dist_cm)
            # All cells along ray except endpoint → free
            for r, c in cells[:-1]:
                new_cells += self._update_cell(r, c, free=True)
            # Last cell → occupied
            if cells:
                r, c = cells[-1]
                new_cells += self._update_cell(r, c, free=False)

        # --- Mark free rays ---
        if free_angles_deg and free_dists_cm:
            for angle_deg, dist_cm in zip(free_angles_deg, free_dists_cm):
                for r, c in self._ray_cells(pose, angle_deg, dist_cm):
                    new_cells += self._update_cell(r, c, free=True)

        # Apply Gaussian inflation around obstacles (for safety margin)
        self._inflate_obstacles()

        # Decay toward prior for dynamic environments
        self._grid += self._decay_rate * (self._lo_prior - self._grid)

        # Mark robot cell as free
        rr, rc = self.world_to_grid(pose.x, pose.y)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                self._update_cell(rr + dr, rc + dc, free=True)

        # Expand grid if near edge
        self._maybe_expand(rr, rc)

        self._explored_cells += new_cells
        return new_cells

    def get_probability(self, row: int, col: int) -> float:
        """Return occupancy probability [0, 1] for a grid cell."""
        if not self._in_bounds(row, col):
            return 0.5
        lo = float(self._grid[row, col])
        return 1.0 / (1.0 + math.exp(-lo))

    def is_obstacle(self, row: int, col: int) -> bool:
        return self.get_probability(row, col) >= self._obs_thresh

    def is_free(self, row: int, col: int) -> bool:
        return (self._known[row, col]
                and self.get_probability(row, col) <= self._free_thresh)

    def is_unknown(self, row: int, col: int) -> bool:
        return not self._known[row, col]

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        col = int(round(wx / self._res)) + self._origin_col
        row = int(round(wy / self._res)) + self._origin_row
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        wx = (col - self._origin_col) * self._res
        wy = (row - self._origin_row) * self._res
        return wx, wy

    def get_frontiers(self, pose: RobotPose,
                      min_size: int = 3) -> List[GridCell]:
        """
        Return frontier cells (free cells adjacent to unknown cells).
        Result is cached and refreshed at most once per second.
        """
        if time.monotonic() - self._last_frontier_update < 1.0:
            return self._frontiers

        robot_r, robot_c = self.world_to_grid(pose.x, pose.y)
        frontiers: List[GridCell] = []
        h, w = self._grid.shape

        # Build probability image for efficiency
        prob = 1.0 / (1.0 + np.exp(-self._grid))
        free_mask    = (prob <= self._free_thresh) & self._known
        unknown_mask = ~self._known

        # Dilate unknown mask to find free cells touching unknown
        kernel = np.ones((3, 3), dtype=np.uint8)
        unknown_dilated = cv2.dilate(unknown_mask.astype(np.uint8),
                                      kernel).astype(bool)
        frontier_mask = free_mask & unknown_dilated

        # Label connected frontier regions
        labeled, n_labels = cv2.connectedComponents(
            frontier_mask.astype(np.uint8))
        for label in range(1, n_labels + 1):
            region = np.argwhere(labeled == label)
            if len(region) < min_size:
                continue
            # Centroid of region
            centroid = region.mean(axis=0)
            frontiers.append(GridCell(int(centroid[0]), int(centroid[1])))

        # Sort by distance from robot
        def dist_to_robot(fc: GridCell) -> float:
            return math.hypot(fc.row - robot_r, fc.col - robot_c)
        frontiers.sort(key=dist_to_robot)

        self._frontiers = frontiers
        self._last_frontier_update = time.monotonic()
        return frontiers

    def get_local_window(
        self,
        pose: RobotPose,
        window_size: int = 7,
    ) -> np.ndarray:
        """
        Return a (window_size, window_size) float32 probability patch
        centred on the robot, suitable as RL state input.
        Values: 0.0=free, 0.5=unknown, 1.0=occupied.
        """
        rr, rc = self.world_to_grid(pose.x, pose.y)
        half = window_size // 2
        patch = np.full((window_size, window_size), 0.5, dtype=np.float32)
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                r, c = rr + dr, rc + dc
                if self._in_bounds(r, c):
                    patch[dr + half, dc + half] = self.get_probability(r, c)
        return patch

    def to_image(self, pose: Optional[RobotPose] = None) -> np.ndarray:
        """Render the grid as a BGR image for visualisation/saving."""
        prob = 1.0 / (1.0 + np.exp(-self._grid))
        img = np.zeros((*self._grid.shape, 3), dtype=np.uint8)
        img[prob > self._obs_thresh]                    = [0, 0, 80]    # dark red = occupied
        img[(prob <= self._obs_thresh)
             & (prob > self._free_thresh) & self._known] = [128, 96, 0] # blue-grey = uncertain
        img[self._known & (prob <= self._free_thresh)]  = [220, 220, 220] # light = free
        img[~self._known]                               = [40, 40, 40]  # dark = unknown

        if pose is not None:
            rr, rc = self.world_to_grid(pose.x, pose.y)
            if self._in_bounds(rr, rc):
                cv2.circle(img, (rc, rr), 3, (0, 255, 0), -1)
                # Draw heading arrow
                dx = int(6 * math.cos(pose.theta))
                dy = int(6 * math.sin(pose.theta))
                cv2.arrowedLine(img, (rc, rr),
                                 (rc + dx, rr + dy),
                                 (0, 255, 100), 1, tipLength=0.4)
        return img

    @property
    def explored_cells(self) -> int:
        return self._explored_cells

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return self._grid.shape

    # ── Internal helpers ──────────────────────────────────────────

    def _ray_cells(
        self,
        pose: RobotPose,
        angle_world_deg: float,
        dist_cm: float,
    ) -> List[Tuple[int, int]]:
        """
        Bresenham ray from robot position to obstacle at given angle/dist.
        Returns list of (row, col) from start to end (inclusive).
        """
        r0, c0 = self.world_to_grid(pose.x, pose.y)
        dist_m = dist_cm / 100.0
        angle_rad = math.radians(angle_world_deg)
        wx = pose.x + dist_m * math.cos(angle_rad)
        wy = pose.y + dist_m * math.sin(angle_rad)
        r1, c1 = self.world_to_grid(wx, wy)

        # Bresenham line
        cells: List[Tuple[int, int]] = []
        dr = abs(r1 - r0); dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        r, c = r0, c0
        if dr > dc:
            err = dr // 2
            while r != r1:
                if self._in_bounds(r, c):
                    cells.append((r, c))
                err -= dc
                if err < 0:
                    c += sc; err += dr
                r += sr
        else:
            err = dc // 2
            while c != c1:
                if self._in_bounds(r, c):
                    cells.append((r, c))
                err -= dr
                if err < 0:
                    r += sr; err += dc
                c += sc
        if self._in_bounds(r1, c1):
            cells.append((r1, c1))
        return cells

    def _update_cell(self, row: int, col: int, free: bool) -> int:
        """Update a single cell's log-odds; return 1 if newly explored."""
        if not self._in_bounds(row, col):
            return 0
        was_unknown = not self._known[row, col]
        if free:
            self._grid[row, col] = np.clip(
                self._grid[row, col] + (self._lo_miss - self._lo_prior),
                self._lo_min, self._lo_max)
        else:
            self._grid[row, col] = np.clip(
                self._grid[row, col] + (self._lo_hit - self._lo_prior),
                self._lo_min, self._lo_max)
        self._known[row, col] = True
        return 1 if was_unknown else 0

    def _inflate_obstacles(self) -> None:
        """Apply Gaussian blur to inflate obstacles (safety margin)."""
        occ_mask = (1.0 / (1.0 + np.exp(-self._grid)) > self._obs_thresh).astype(np.float32)
        inflated = gaussian_filter(occ_mask, sigma=self._blur_sigma)
        inflate_lo = _log_odds(np.clip(inflated * 0.9, 0.01, 0.99))
        overlay = inflated > 0.3
        self._grid[overlay] = np.clip(
            np.maximum(self._grid[overlay], inflate_lo[overlay] * 0.5),
            self._lo_min, self._lo_max)

    def _maybe_expand(self, robot_row: int, robot_col: int) -> None:
        """Expand the grid if the robot is too close to any edge."""
        h, w = self._grid.shape
        pad = self._exp_thresh
        amt = self._exp_amount
        expand = [0, 0, 0, 0]   # top, bottom, left, right
        if robot_row < pad:       expand[0] = amt
        if robot_row > h - pad:   expand[1] = amt
        if robot_col < pad:       expand[2] = amt
        if robot_col > w - pad:   expand[3] = amt

        if any(expand):
            top, bot, left, right = expand
            new_h = h + top + bot
            new_w = w + left + right
            new_grid  = np.full((new_h, new_w), self._lo_prior, dtype=np.float32)
            new_known = np.zeros((new_h, new_w), dtype=bool)
            new_grid[top:top+h, left:left+w]  = self._grid
            new_known[top:top+h, left:left+w] = self._known
            self._grid  = new_grid
            self._known = new_known
            self._origin_row += top
            self._origin_col += left
            logger.info("[Map] Expanded to %dx%d", new_h, new_w)

    def _in_bounds(self, r: int, c: int) -> bool:
        h, w = self._grid.shape
        return 0 <= r < h and 0 <= c < w
