"""
perception/obstacle_detector.py
Obstacle detection using depth + visual cues for Yahboom Raspbot V2.

Two approaches are combined:
  1. Depth-map thresholding  – fast, frame-by-frame
  2. Object detection (YOLOv5n / MobileNet-SSD) – richer semantic info

Each detected obstacle is described by an ObstacleBox namedtuple
containing its bounding box (normalised 0-1), confidence, class label,
estimated distance (cm), and angular offset from camera centre (deg).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ObstacleBox:
    x1: float          # normalised [0, 1]
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str
    dist_cm: float     # estimated distance from ultrasonic / depth fusion
    angle_deg: float   # horizontal angle from optical axis (+ = right)


class ObstacleDetector:
    """
    Parameters
    ----------
    cfg : dict
        The ``perception.obstacle`` sub-dict from robot_config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._conf_threshold = float(cfg.get("confidence_threshold", 0.50))
        self._nms_threshold  = float(cfg.get("nms_iou_threshold",   0.45))
        self._min_area       = int(cfg.get("min_pixel_area", 500))
        self._danger_zone_px = int(cfg.get("danger_zone_px", 80))
        self._model = None
        self._yolo  = None

        # Try loading YOLOv5 nano (optional; depth fallback always works)
        weights = cfg.get("weights", "models/yolov5n.pt")
        self._classes_of_interest: Optional[List[int]] = cfg.get("classes_of_interest")
        self._load_yolo(weights)

    # ── Initialisation ────────────────────────────────────────────

    def _load_yolo(self, weights: str) -> None:
        import os
        if not os.path.exists(weights):
            logger.info("[Obstacles] Weights %s not found – depth-only mode", weights)
            return
        try:
            import torch
            self._yolo = torch.hub.load("ultralytics/yolov5", "custom",
                                         path=weights, trust_repo=True,
                                         verbose=False)
            self._yolo.conf = self._conf_threshold
            self._yolo.iou  = self._nms_threshold
            logger.info("[Obstacles] YOLOv5n loaded from %s", weights)
        except Exception as exc:
            logger.warning("[Obstacles] YOLOv5 unavailable (%s) – depth-only", exc)

    # ── Public API ────────────────────────────────────────────────

    def detect(
        self,
        bgr_frame: np.ndarray,
        depth_map: np.ndarray,          # (H_d, W_d) float32 normalised
        ultrasonic_cm: float = 200.0,   # front-centre reading for fusion
        camera_fov_deg: float = 62.0,   # horizontal FOV
    ) -> List[ObstacleBox]:
        """
        Detect obstacles, returning a list of ObstacleBox instances.

        Parameters
        ----------
        bgr_frame    : full-resolution BGR frame
        depth_map    : downsampled depth map from DepthEstimator
        ultrasonic_cm: ultrasonic front-centre reading for scale calibration
        camera_fov_deg: horizontal field of view of the camera lens
        """
        boxes: List[ObstacleBox] = []

        # --- 1. Depth-based fast detection ---
        depth_boxes = self._detect_from_depth(bgr_frame, depth_map,
                                               ultrasonic_cm, camera_fov_deg)
        boxes.extend(depth_boxes)

        # --- 2. YOLO semantic detection ---
        if self._yolo is not None:
            yolo_boxes = self._detect_yolo(bgr_frame, depth_map,
                                            ultrasonic_cm, camera_fov_deg)
            # Merge – prefer YOLO detections in overlapping regions
            boxes = self._merge(boxes, yolo_boxes)

        return boxes

    def nearest_obstacle_cm(self, boxes: List[ObstacleBox]) -> float:
        """Return the range of the closest detected obstacle."""
        if not boxes:
            return float("inf")
        return min(b.dist_cm for b in boxes)

    # ── Detection helpers ─────────────────────────────────────────

    def _detect_from_depth(
        self,
        bgr: np.ndarray,
        depth: np.ndarray,
        us_cm: float,
        fov_deg: float,
    ) -> List[ObstacleBox]:
        """
        Find contiguous near regions in the depth map.
        Near ≡ depth value < threshold (close objects are low in map).
        """
        boxes: List[ObstacleBox] = []
        h_frame, w_frame = bgr.shape[:2]
        h_d, w_d = depth.shape

        # Calibrate: use ultrasonic reading to anchor near threshold
        # 200 cm → depth_value ~ 0.5 (linear assumption)
        us_norm = min(1.0, us_cm / 400.0)
        near_threshold = max(0.10, 1.0 - us_norm + 0.10)

        near_mask = (depth < near_threshold).astype(np.uint8) * 255
        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(near_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Scale area to frame pixels
            area_px = area * (w_frame * h_frame) / (w_d * h_d)
            if area_px < self._min_pixel_size():
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Normalise to [0,1] in depth map space → convert to frame space
            cx_n = (x + w / 2) / w_d
            cy_n = (y + h / 2) / h_d
            # Estimate depth value in this region
            d_val = float(depth[y:y+h, x:x+w].mean())
            dist_cm = max(5.0, (1.0 - d_val) * 400.0)
            angle = (cx_n - 0.5) * fov_deg
            boxes.append(ObstacleBox(
                x1=x / w_d, y1=y / h_d,
                x2=(x+w) / w_d, y2=(y+h) / h_d,
                confidence=0.70,
                label="obstacle",
                dist_cm=dist_cm,
                angle_deg=angle,
            ))
        return boxes

    def _detect_yolo(
        self,
        bgr: np.ndarray,
        depth: np.ndarray,
        us_cm: float,
        fov_deg: float,
    ) -> List[ObstacleBox]:
        boxes: List[ObstacleBox] = []
        try:
            import torch
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = self._yolo(rgb, size=320)
            h_frame, w_frame = bgr.shape[:2]
            h_d, w_d = depth.shape

            for *xyxy, conf, cls_id in results.xyxy[0].cpu().numpy():
                cls_int = int(cls_id)
                if (self._classes_of_interest
                        and cls_int not in self._classes_of_interest):
                    continue
                x1n, y1n = float(xyxy[0])/w_frame, float(xyxy[1])/h_frame
                x2n, y2n = float(xyxy[2])/w_frame, float(xyxy[3])/h_frame
                cx_n = (x1n + x2n) / 2
                cy_n = (y1n + y2n) / 2

                # Depth at the bounding-box centre
                dx = int(cx_n * w_d)
                dy = int(cy_n * h_d)
                d_val = float(depth[
                    max(0, dy-1):min(h_d, dy+2),
                    max(0, dx-1):min(w_d, dx+2)
                ].mean())
                dist_cm = max(5.0, (1.0 - d_val) * 400.0)
                angle = (cx_n - 0.5) * fov_deg

                label = results.names[cls_int] if results.names else str(cls_int)
                boxes.append(ObstacleBox(
                    x1=x1n, y1=y1n, x2=x2n, y2=y2n,
                    confidence=float(conf),
                    label=label,
                    dist_cm=dist_cm,
                    angle_deg=angle,
                ))
        except Exception as exc:
            logger.debug("[Obstacles] YOLO inference error: %s", exc)
        return boxes

    @staticmethod
    def _merge(
        depth_boxes: List[ObstacleBox],
        yolo_boxes: List[ObstacleBox],
        iou_threshold: float = 0.30,
    ) -> List[ObstacleBox]:
        """NMS-style merge: keep YOLO box if it overlaps a depth box."""
        if not yolo_boxes:
            return depth_boxes
        if not depth_boxes:
            return yolo_boxes

        merged = list(yolo_boxes)
        for db in depth_boxes:
            overlaps = any(
                _iou(db, yb) > iou_threshold for yb in yolo_boxes
            )
            if not overlaps:
                merged.append(db)
        return merged

    def _min_pixel_size(self) -> int:
        return self._min_area


def _iou(a: ObstacleBox, b: ObstacleBox) -> float:
    ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2-ix1) * (iy2-iy1)
    union = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter
    return inter / union if union > 1e-8 else 0.0
