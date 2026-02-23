"""
perception/depth_estimator.py
Monocular depth estimation for Yahboom Raspbot V2.

Primary model : MiDaS (midas_small or midas_v21_small_256)
                 – runs in < 100 ms on RPi 5 with ARM NEON
Fallback      : gradient-based proxy using vertical image gradient
                 (faster, ~5 ms, less accurate)

Output: a (16, 16) float32 numpy array, normalised 0-1 where
        1.0 → furthest and 0.0 → closest / unknown.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MIDAS_MODELS = {
    "midas_small":   "MiDaS_small",
    "midas_v21":     "MiDaS_v21_small_256",
}


class DepthEstimator:
    """
    Parameters
    ----------
    cfg : dict
        The ``perception.depth`` sub-dict from robot_config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._model_name = cfg.get("model", "none")
        self._out_h, self._out_w = cfg.get("output_size", [16, 16])
        self._normalize = cfg.get("normalize", True)
        self._model = None
        self._transform = None
        self._device = "cpu"

        if self._model_name != "none":
            self._load_midas()

    # ── Initialisation ────────────────────────────────────────────

    def _load_midas(self) -> None:
        try:
            import torch
            torch_hub_name = _MIDAS_MODELS.get(self._model_name, "MiDaS_small")
            logger.info("[Depth] Loading %s ...", torch_hub_name)
            self._model = torch.hub.load("intel-isl/MiDaS", torch_hub_name,
                                          trust_repo=True)
            self._model.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                         trust_repo=True)
            self._transform = transforms.small_transform
            self._torch = torch
            logger.info("[Depth] MiDaS loaded on %s", self._device)
        except Exception as exc:
            logger.warning("[Depth] MiDaS unavailable (%s) -> gradient proxy", exc)
            self._model_name = "none"

    # ── Public API ────────────────────────────────────────────────

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from a single BGR frame.

        Returns
        -------
        depth : np.ndarray, shape (out_h, out_w), dtype float32
            Values in [0, 1]; higher = further away.
        """
        if self._model is not None:
            return self._midas_inference(bgr_frame)
        return self._gradient_proxy(bgr_frame)

    # ── Inference paths ───────────────────────────────────────────

    def _midas_inference(self, bgr: np.ndarray) -> np.ndarray:
        try:
            import torch
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inp = self._transform(rgb).to(self._device)

            with torch.no_grad():
                raw = self._model(inp)
                raw = torch.nn.functional.interpolate(
                    raw.unsqueeze(1),
                    size=(self._out_h, self._out_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()

            if self._normalize:
                mn, mx = raw.min(), raw.max()
                if mx - mn > 1e-6:
                    raw = (raw - mn) / (mx - mn)

            return raw.astype(np.float32)
        except Exception as exc:
            logger.debug("[Depth] MiDaS inference error: %s", exc)
            return self._gradient_proxy(bgr)

    def _gradient_proxy(self, bgr: np.ndarray) -> np.ndarray:
        """
        Cheap proxy: vertical position correlates with distance on flat
        indoor floors.  Lower in frame → nearer.  Combine with Sobel
        gradient magnitude to detect depth discontinuities.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape

        # Vertical distance ramp (bottom = 0 = near, top = 1 = far)
        ramp = np.linspace(1.0, 0.0, h).reshape(h, 1) * np.ones((1, w))

        # Suppress ramp where strong edges indicate obstacles
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        edge_mag = cv2.magnitude(sobel_x, sobel_y)
        edge_norm = cv2.normalize(edge_mag, None, 0, 1, cv2.NORM_MINMAX)

        depth = ramp * (1.0 - 0.5 * edge_norm)

        # Downsample to output resolution
        depth_small = cv2.resize(depth.astype(np.float32),
                                  (self._out_w, self._out_h),
                                  interpolation=cv2.INTER_AREA)
        return depth_small
