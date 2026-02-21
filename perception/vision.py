"""
perception/vision.py
Unified visual perception pipeline for Yahboom Raspbot V2.

Orchestrates DepthEstimator, ObstacleDetector, FloorDetector, and a
MobileNetV2 feature extractor, delivering a single PerceptionResult
on every call to process().  Designed to run on the RPi 5 CPU at
~10 fps with lightweight models.

Also exposes a draw_debug() helper for overlaying all detections on a
BGR frame for remote monitoring / logging.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .depth_estimator  import DepthEstimator
from .obstacle_detector import ObstacleDetector, ObstacleBox
from .floor_detector   import FloorDetector, FloorMap

logger = logging.getLogger(__name__)


@dataclass
class PerceptionResult:
    timestamp: float
    # Raw outputs
    depth_map: np.ndarray           # (16, 16) float32, 0-1
    obstacle_boxes: List[ObstacleBox]
    floor_map: FloorMap
    # Feature vector for RL state
    visual_features: np.ndarray     # (feature_dim,) float32
    # Convenience
    nearest_obstacle_cm: float
    floor_clear: bool               # True if floor width > 40%
    processing_ms: float


class VisionPipeline:
    """
    Parameters
    ----------
    cfg : dict
        The ``perception`` sub-dict from robot_config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._vis_cfg  = cfg.get("vision", {})
        self._inp_size = tuple(self._vis_cfg.get("input_size", [224, 224]))
        self._feat_dim = int(self._vis_cfg.get("feature_dim", 128))
        self._device   = self._vis_cfg.get("device", "cpu")

        self._depth_est  = DepthEstimator(cfg.get("depth",    {}))
        self._obstacle   = ObstacleDetector(cfg.get("objects",  {}))
        # Pass obstacle cfg for confidence/threshold settings too
        self._obstacle_vis = ObstacleDetector(cfg.get("obstacle", {}))
        self._floor      = FloorDetector(cfg.get("floor",     {}))
        self._extractor  = self._build_extractor()

    # ── Feature extractor ─────────────────────────────────────────

    def _build_extractor(self):
        """
        Load MobileNetV2 truncated before the classifier.
        Returns None if torch is unavailable → random projection fallback.
        """
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            base = models.mobilenet_v2(weights="IMAGENET1K_V1")
            # Keep feature layers, replace classifier with projection
            features = base.features
            pool     = torch.nn.AdaptiveAvgPool2d((1, 1))
            proj     = torch.nn.Linear(1280, self._feat_dim)
            extractor = torch.nn.Sequential(features, pool,
                                             torch.nn.Flatten(), proj).to(self._device)
            extractor.eval()

            self._transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])
            logger.info("[Vision] MobileNetV2 extractor loaded, feat_dim=%d",
                        self._feat_dim)
            return extractor
        except Exception as exc:
            logger.warning("[Vision] Torch unavailable (%s) – random projection", exc)
            # Reproducible random projection matrix (640×480 BGR → feat_dim)
            rng = np.random.default_rng(42)
            flattened = self._inp_size[0] * self._inp_size[1] * 3
            self._rand_proj = rng.standard_normal(
                (flattened, self._feat_dim)).astype(np.float32) * 0.01
            return None

    # ── Public API ────────────────────────────────────────────────

    def process(
        self,
        bgr_frame: np.ndarray,
        ultrasonic_cm: float = 200.0,
    ) -> PerceptionResult:
        """
        Run the full perception pipeline on a single BGR frame.

        Parameters
        ----------
        bgr_frame     : camera frame (BGR, any resolution)
        ultrasonic_cm : front-centre ultrasonic reading for fusion
        """
        t0 = time.monotonic()

        # Resize to processing resolution
        proc = cv2.resize(bgr_frame, (self._inp_size[1], self._inp_size[0]))

        # --- Depth ---
        depth_map = self._depth_est.estimate(proc)

        # --- Obstacles ---
        obs_boxes = self._obstacle.detect(proc, depth_map, ultrasonic_cm)
        nearest   = self._obstacle.nearest_obstacle_cm(obs_boxes)

        # --- Floor ---
        floor_map = self._floor.detect(proc)

        # --- Visual features ---
        features = self._extract_features(proc)

        ms = (time.monotonic() - t0) * 1000.0

        return PerceptionResult(
            timestamp=time.monotonic(),
            depth_map=depth_map,
            obstacle_boxes=obs_boxes,
            floor_map=floor_map,
            visual_features=features,
            nearest_obstacle_cm=nearest,
            floor_clear=(floor_map.floor_width_frac > 0.40),
            processing_ms=ms,
        )

    def draw_debug(self, bgr_frame: np.ndarray,
                   result: PerceptionResult) -> np.ndarray:
        """Overlay detections on a copy of the frame and return it."""
        out = bgr_frame.copy()
        h, w = out.shape[:2]

        # Draw obstacle boxes
        for box in result.obstacle_boxes:
            x1i = int(box.x1 * w); y1i = int(box.y1 * h)
            x2i = int(box.x2 * w); y2i = int(box.y2 * h)
            color = (0, 0, 255) if box.dist_cm < 60 else (0, 165, 255)
            cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 2)
            cv2.putText(out, f"{box.label} {box.dist_cm:.0f}cm",
                        (x1i, max(0, y1i - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw floor boundary line
        hy = result.floor_map.horizon_line_px
        cv2.line(out, (0, hy), (w, hy), (0, 255, 0), 1)

        # Depth map overlay (top-right corner)
        dm = result.depth_map
        dm_vis = (dm * 255).astype(np.uint8)
        dm_col = cv2.applyColorMap(dm_vis, cv2.COLORMAP_INFERNO)
        dm_big = cv2.resize(dm_col, (80, 80))
        out[4:84, w-84:w-4] = dm_big

        # HUD
        cv2.putText(out,
                    f"Near:{result.nearest_obstacle_cm:.0f}cm  "
                    f"t:{result.processing_ms:.0f}ms",
                    (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return out

    # ── Internal ─────────────────────────────────────────────────

    def _extract_features(self, bgr_resized: np.ndarray) -> np.ndarray:
        if self._extractor is not None:
            try:
                import torch
                rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil = Image.fromarray(rgb)
                tensor = self._transform(pil).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = self._extractor(tensor).squeeze().cpu().numpy()
                return feat.astype(np.float32)
            except Exception as exc:
                logger.debug("[Vision] Feature extraction error: %s", exc)

        # Fallback: random projection of flattened grayscale
        gray = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        flat = gray.flatten()
        if len(flat) != self._rand_proj.shape[0]:
            # Resize to expected size
            gray = cv2.resize(gray, (self._inp_size[1], self._inp_size[0]))
            flat = gray.flatten()
        proj = flat @ self._rand_proj
        return proj.astype(np.float32)
