"""
navigation/path_planner.py
A* path planner operating on the OccupancyGrid.

Features:
  • Inflated safety margins around obstacles (configurable in cells).
  • Diagonal movement with correct √2 cost.
  • Path smoothing via iterative shortcutting.
  • Returns world-frame waypoint list (metres) for the controller.
"""
from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from mapping.occupancy_grid import OccupancyGrid, RobotPose

logger = logging.getLogger(__name__)


@dataclass
class Path:
    waypoints: List[Tuple[float, float]]   # (x, y) world metres
    length_m: float
    n_cells: int

    @property
    def empty(self) -> bool:
        return len(self.waypoints) == 0


class AStarPlanner:
    """
    Parameters
    ----------
    cfg : dict
        The ``navigation.planner`` sub-dict from robot_config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._heuristic    = cfg.get("heuristic", "euclidean")
        self._safety       = int(cfg.get("safety_margin_cells", 2))
        self._max_length   = int(cfg.get("max_plan_length", 500))
        self._smooth       = bool(cfg.get("path_smoothing", True))
        self._smooth_iters = int(cfg.get("smoothing_iterations", 3))

    # ── Public API ────────────────────────────────────────────────

    def plan(
        self,
        grid: OccupancyGrid,
        start: RobotPose,
        goal_world: Tuple[float, float],
    ) -> Optional[Path]:
        """
        Find a path from ``start`` to ``goal_world``.

        Returns None if no path exists or goal is occupied.
        """
        sr, sc = grid.world_to_grid(start.x, start.y)
        gr, gc = grid.world_to_grid(*goal_world)

        if not grid._in_bounds(gr, gc):
            logger.debug("[Planner] Goal out of bounds (%d, %d)", gr, gc)
            return None
        if grid.is_obstacle(gr, gc):
            logger.debug("[Planner] Goal is occupied – cannot plan")
            return None

        cell_path = self._astar(grid, (sr, sc), (gr, gc))
        if cell_path is None:
            return None
        if len(cell_path) > self._max_length:
            logger.warning("[Planner] Path too long (%d > %d cells) – truncating",
                           len(cell_path), self._max_length)
            cell_path = cell_path[:self._max_length]

        if self._smooth:
            cell_path = self._smooth_path(cell_path, grid)

        waypoints = [grid.grid_to_world(r, c) for r, c in cell_path]

        # Path length
        length_m = 0.0
        for i in range(1, len(waypoints)):
            dx = waypoints[i][0] - waypoints[i-1][0]
            dy = waypoints[i][1] - waypoints[i-1][1]
            length_m += math.hypot(dx, dy)

        return Path(waypoints=waypoints, length_m=length_m,
                     n_cells=len(cell_path))

    # ── A* core ───────────────────────────────────────────────────

    def _astar(
        self,
        grid: OccupancyGrid,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:

        # 8-connected neighbours: (dr, dc, cost)
        neighbors = [
            (-1,  0, 1.0), ( 1,  0, 1.0),
            ( 0, -1, 1.0), ( 0,  1, 1.0),
            (-1, -1, math.sqrt(2)), (-1,  1, math.sqrt(2)),
            ( 1, -1, math.sqrt(2)), ( 1,  1, math.sqrt(2)),
        ]

        open_set: list = []       # min-heap of (f, g, node)
        g_cost: Dict[Tuple[int, int], float] = {start: 0.0}
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        h0 = self._h(start, goal)
        heapq.heappush(open_set, (h0, 0.0, start))

        while open_set:
            f, g, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct(parent, current)
            if g > g_cost.get(current, float("inf")):
                continue   # outdated entry

            cr, cc = current
            for dr, dc, move_cost in neighbors:
                nr, nc = cr + dr, cc + dc
                if not grid._in_bounds(nr, nc):
                    continue
                if self._is_blocked(grid, nr, nc):
                    continue
                new_g = g + move_cost
                if new_g < g_cost.get((nr, nc), float("inf")):
                    g_cost[(nr, nc)] = new_g
                    parent[(nr, nc)] = current
                    f_new = new_g + self._h((nr, nc), goal)
                    heapq.heappush(open_set, (f_new, new_g, (nr, nc)))

        return None   # no path

    def _is_blocked(self, grid: OccupancyGrid, r: int, c: int) -> bool:
        """Return True if cell is obstacle or within safety margin."""
        if grid.is_obstacle(r, c):
            return True
        # Check safety margin neighbours
        for dr in range(-self._safety, self._safety + 1):
            for dc in range(-self._safety, self._safety + 1):
                if grid._in_bounds(r+dr, c+dc) and grid.is_obstacle(r+dr, c+dc):
                    return True
        return False

    def _h(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dr = abs(a[0] - b[0]); dc = abs(a[1] - b[1])
        if self._heuristic == "manhattan":
            return dr + dc
        elif self._heuristic == "diagonal":
            return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)
        return math.hypot(dr, dc)   # euclidean (default)

    @staticmethod
    def _reconstruct(
        parent: Dict,
        node: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    # ── Path smoothing ────────────────────────────────────────────

    def _smooth_path(
        self,
        path: List[Tuple[int, int]],
        grid: OccupancyGrid,
    ) -> List[Tuple[int, int]]:
        """
        Iterative shortcutting: try to connect waypoints directly skipping
        intermediate cells, provided the straight-line segment is collision-free.
        """
        for _ in range(self._smooth_iters):
            if len(path) <= 2:
                break
            new_path = [path[0]]
            i = 0
            while i < len(path) - 1:
                # Find the furthest point reachable via line-of-sight
                j = len(path) - 1
                while j > i + 1:
                    if self._line_of_sight(grid, path[i], path[j]):
                        break
                    j -= 1
                new_path.append(path[j])
                i = j
            path = new_path
        return path

    def _line_of_sight(
        self,
        grid: OccupancyGrid,
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> bool:
        """Bresenham line-of-sight check between two grid cells."""
        r0, c0 = a; r1, c1 = b
        dr = abs(r1 - r0); dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        r, c = r0, c0
        if dr > dc:
            err = dr // 2
            while r != r1:
                if self._is_blocked(grid, r, c):
                    return False
                err -= dc
                if err < 0:
                    c += sc; err += dr
                r += sr
        else:
            err = dc // 2
            while c != c1:
                if self._is_blocked(grid, r, c):
                    return False
                err -= dr
                if err < 0:
                    r += sr; err += dc
                c += sc
        return True
