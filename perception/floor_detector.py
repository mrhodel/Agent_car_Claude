"""
perception/floor_detector.py
Floor and traversable-area boundary detection.

Two methods are supported (configurable via robot_config.yaml):
  • edge_color  – HSV colour segmentation + Canny edge boundary line
  • homography  – IPM (Inverse Perspective Mapping) perspective warp,
                  useful when the floor has a distinct uniform colour

Returns a FloorMap dataclass with:
  - traversable_mask : (H, W) uint8  (255 = free floor, 0 = unknown/obstacle)
  - boundary_y_frac  : float         fraction of frame where floor ends (top)
  - floor_width_frac : float         estimated traversable width (0-1)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FloorMap:
    traversable_mask: np.ndarray      # (H, W) uint8
    boundary_y_frac: float            # 0=top, 1=bottom; floor starts below this
    floor_width_frac: float           # 0–1, estimated clear width fraction
    horizon_line_px: int              # pixel row of estimated horizon


class FloorDetector:
    """
    Parameters
    ----------
    cfg : dict
        The ``perception.floor`` sub-dict from robot_config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._method = cfg.get("method", "edge_color")
        self._hsv_low  = np.array(cfg.get("hsv_floor_low",  [0,  0,  80]))
        self._hsv_high = np.array(cfg.get("hsv_floor_high", [180, 50, 255]))
        self._boundary_rows = float(cfg.get("boundary_rows", 0.55))
        # IPM homography corners (normalised, set for ~30° camera tilt)
        self._src_pts = np.float32([[0.0, 1.0], [1.0, 1.0],
                                    [0.7, 0.5], [0.3, 0.5]])
        self._dst_pts = np.float32([[0.1, 1.0], [0.9, 1.0],
                                    [0.9, 0.0], [0.1, 0.0]])

    # ── Public API ────────────────────────────────────────────────

    def detect(self, bgr_frame: np.ndarray) -> FloorMap:
        h, w = bgr_frame.shape[:2]

        if self._method == "homography":
            return self._ipm_method(bgr_frame, h, w)
        return self._edge_color_method(bgr_frame, h, w)

    # ── Method implementations ────────────────────────────────────

    def _edge_color_method(self, bgr: np.ndarray, h: int, w: int) -> FloorMap:
        """
        1. HSV colour-range mask for floor texture/colour.
        2. Canny edges to find walls and obstacles.
        3. Combine: floor mask AND NOT edge mask.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self._hsv_low, self._hsv_high)

        # Only look at lower part of frame (floor region)
        horizon_px = int(h * self._boundary_rows)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[horizon_px:, :] = 255
        color_mask = cv2.bitwise_and(color_mask, roi_mask)

        # Edge detection to identify boundaries
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8))

        # Traversable = floor colour AND no strong edge
        traversable = cv2.bitwise_and(color_mask,
                                       cv2.bitwise_not(edges_dilated))

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        traversable = cv2.morphologyEx(traversable, cv2.MORPH_CLOSE, kernel)
        traversable = cv2.morphologyEx(traversable, cv2.MORPH_OPEN, kernel)

        # Estimate floor width at the bottom of the frame
        bottom_strip = traversable[int(0.9 * h):h, :]
        floor_cols = np.count_nonzero(bottom_strip.sum(axis=0))
        floor_width_frac = floor_cols / w

        return FloorMap(
            traversable_mask=traversable,
            boundary_y_frac=self._boundary_rows,
            floor_width_frac=floor_width_frac,
            horizon_line_px=horizon_px,
        )

    def _ipm_method(self, bgr: np.ndarray, h: int, w: int) -> FloorMap:
        """
        Apply Inverse Perspective Mapping (bird's-eye view) then colour-segment.
        """
        src = (self._src_pts * np.array([w, h])).astype(np.float32)
        dst = (self._dst_pts * np.array([w, h])).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        bird = cv2.warpPerspective(bgr, M, (w, h))

        hsv = cv2.cvtColor(bird, cv2.COLOR_BGR2HSV)
        bev_mask = cv2.inRange(hsv, self._hsv_low, self._hsv_high)

        # Warp mask back to original perspective
        M_inv = cv2.getPerspectiveTransform(dst, src)
        traversable = cv2.warpPerspective(bev_mask, M_inv, (w, h))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        traversable = cv2.morphologyEx(traversable, cv2.MORPH_CLOSE, kernel)

        horizon_px = int(h * self._boundary_rows)
        bottom_strip = traversable[int(0.9 * h):h, :]
        floor_cols = np.count_nonzero(bottom_strip.sum(axis=0))
        floor_width_frac = floor_cols / w

        return FloorMap(
            traversable_mask=traversable,
            boundary_y_frac=self._boundary_rows,
            floor_width_frac=floor_width_frac,
            horizon_line_px=horizon_px,
        )
