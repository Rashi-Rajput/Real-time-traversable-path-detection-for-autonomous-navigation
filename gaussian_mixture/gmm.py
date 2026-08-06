#!/usr/bin/env python3


import time
import argparse
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque
import logging
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("gmm_fast")

# ─────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    video_path: str = "videos/vid7.mp4"

    # Resolution - balance speed vs accuracy
    max_proc_width: int  = 480   # lower = faster GMM scoring
    gmm_scale: float     = 0.50  # additional downscale for GMM only

    # Ground plane geometry
    horizon_frac: float  = 0.42
    roi_top_frac: float  = 0.48
    seed_top_frac: float = 0.70
    seed_bot_frac: float = 0.90
    seed_cx_frac: float  = 0.28   # half-width of seed trapezoid

    # Superpixel settings (SLIC)
    use_superpixels: bool = True
    slic_region_size: int = 16    # smaller = more detail, slower
    slic_ruler: float     = 10.0

    # GMM
    gmm_k: int            = 5     # fixed k (skip BIC for speed)
    gmm_k_range: Tuple[int, int] = (3, 7)
    use_bic: bool         = False  # True = accurate, False = fast
    gmm_covariance: str   = "diag" # "diag" >> "full" in speed
    gmm_reg_covar: float  = 1e-3
    gmm_max_iter: int     = 100
    gmm_n_init: int       = 1
    random_seed: int      = 42
    max_fit_samples: int  = 6000

    # Thresholding
    use_otsu_threshold: bool  = True   # auto threshold
    threshold_percentile: float = 8.0
    threshold_margin: float     = 0.8

    # Online adaptation
    adapt_every_n_frames: int   = 45
    adapt_history_frames: int   = 6
    min_seed_pixels: int        = 150

    # Optical flow temporal consistency
    use_optical_flow: bool  = True
    flow_blend_alpha: float = 0.55  # weight on flow-propagated mask

    # Score EMA (fallback if no optical flow)
    score_ema_alpha: float = 0.55
    mask_ema_alpha: float  = 0.65

    # CLAHE (illumination normalisation)
    clahe_clip: float  = 2.0
    clahe_grid: int    = 8

    # Post-processing
    morph_open_iter: int  = 1
    morph_close_iter: int = 2
    morph_kernel: int     = 5
    min_region_area: int  = 600
    top_n_regions: int    = 3

    
    path_step: int         = 5
    path_smooth_window: int = 7
    min_path_points: int    = 5

    # Display
    show_seed_region: bool = True
    show_score_map: bool   = False

# ─────────────────────────────────────────────────────────────
#  CLAHE Pre-processor (illumination invariance)
# ─────────────────────────────────────────────────────────────

class Preprocessor:
    def __init__(self, cfg: Config):
        self.clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)
        )

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ─────────────────────────────────────────────────────────────
#  Cached spatial grid
# ─────────────────────────────────────────────────────────────

class SpatialGridCache:

    def __init__(self):
        self._cache: dict = {}

    def get(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (h, w)
        if key not in self._cache:
            row_map = (np.arange(h, dtype=np.float32) / h).reshape(h, 1)
            row_map = np.broadcast_to(row_map, (h, w)).copy()
            col_map = np.abs(np.arange(w, dtype=np.float32) - w / 2) / (w / 2)
            col_map = np.broadcast_to(col_map.reshape(1, w), (h, w)).copy()
            self._cache[key] = (row_map, col_map)
        return self._cache[key]

_SPATIAL_CACHE = SpatialGridCache()

# ─────────────────────────────────────────────────────────────
#  LBP Texture (fast, rotation-invariant)
# ─────────────────────────────────────────────────────────────

def lbp_fast(gray: np.ndarray) -> np.ndarray:
    """
    Uniform LBP using bit-operations. ~10x faster than skimage.
    Returns float32 in [0,1].
    """
    g = gray.astype(np.float32)

    # 8 neighbours via shifts
    neighbours = [
        np.roll(np.roll(g,  1, 0),  1, 1),  # top-left
        np.roll(g,  1, 0),                   # top
        np.roll(np.roll(g,  1, 0), -1, 1),  # top-right
        np.roll(g,  0, 0 ) * 0 + np.roll(g, -1, 1),  # right (inline)
        np.roll(np.roll(g, -1, 0), -1, 1),  # bot-right
        np.roll(g, -1, 0),                   # bottom
        np.roll(np.roll(g, -1, 0),  1, 1),  # bot-left
        np.roll(g,  1, 1),                   # left
    ]

    # Safer neighbour access using shift
    def shift(arr, dr, dc):
        return np.roll(np.roll(arr, dr, axis=0), dc, axis=1)

    n = [
        shift(g,  1,  1), shift(g,  1, 0), shift(g,  1, -1),
        shift(g,  0, -1), shift(g, -1, -1), shift(g, -1, 0),
        shift(g, -1,  1), shift(g,  0,  1),
    ]

    code = np.zeros_like(g, dtype=np.uint8)
    for i, nb in enumerate(n):
        code += ((nb >= g).astype(np.uint8) << i)

    return (code.astype(np.float32) / 255.0)

# ─────────────────────────────────────────────────────────────
#  Feature extraction  (fast, 14 dims)
# ─────────────────────────────────────────────────────────────

FEAT_DIM = 14

def extract_features(
    img_bgr: np.ndarray,
    spatial_cache: SpatialGridCache,
) -> np.ndarray:
    """
    Returns float32 (H, W, FEAT_DIM).
    Speed optimisations:
    - Single cvtColor pass for HSV, LAB
    - Pre-cached spatial grids
    - LBP replaces slow box-filter std-dev
    - Avoid redundant copies
    """
    h, w = img_bgr.shape[:2]

    # ── colour ─────────────────────────────────────────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bgr = img_bgr.astype(np.float32)

    # HSV circular hue encoding + saturation + value
    H_rad  = hsv[:, :, 0] * (np.pi / 90.0)   # [0,180] → [0,2π]
    S      = hsv[:, :, 1] * (1.0 / 255.0)
    V      = hsv[:, :, 2] * (1.0 / 255.0)
    hsin_s = np.sin(H_rad) * S
    hcos_s = np.cos(H_rad) * S

    # LAB normalised
    L_n = lab[:, :, 0] * (1.0 / 255.0)
    a_n = lab[:, :, 1] * (1.0 / 255.0)
    b_n = lab[:, :, 2] * (1.0 / 255.0)

    # Chromaticity (illumination-robust)
    bgr_sum = bgr.sum(axis=2) + 1e-6
    b_chr   = bgr[:, :, 0] / bgr_sum
    g_chr   = bgr[:, :, 1] / bgr_sum

    # ── texture ────────────────────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # LBP (fast texture)
    lbp = lbp_fast(gray.astype(np.float32))

    # Gradient magnitude (Scharr is more accurate than Sobel at ksize=3)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx * gx + gy * gy)
    # Normalise in ROI only to avoid sky pulling down scale
    grad_max = np.percentile(grad, 99) + 1e-6
    grad = np.clip(grad / grad_max, 0, 1).astype(np.float32)

    # Laplacian (edge sharpness)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    lap_max = np.percentile(lap, 99) + 1e-6
    lap = np.clip(lap / lap_max, 0, 1).astype(np.float32)

    # ── spatial priors (cached) ────────────────────────────
    row_map, col_map = spatial_cache.get(h, w)

    # ── stack ─────────────────────────────────────────────
    feat = np.stack([
        hsin_s, hcos_s, S, V,      # 4 - hue + sat + val
        L_n, a_n, b_n,             # 3 - perceptual colour
        b_chr, g_chr,              # 2 - illumination-robust chroma
        lbp,                       # 1 - texture
        grad,                      # 1 - gradient
        lap,                       # 1 - edge
        row_map,                   # 1 - perspective prior
        col_map,                   # 1 - symmetry prior
    ], axis=-1)

    return feat.astype(np.float32)

def flat_features(
    feat_map: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    h, w, d = feat_map.shape
    flat = feat_map.reshape(-1, d)
    if mask is not None:
        flat = flat[mask.reshape(-1) > 0]
    return flat

# ─────────────────────────────────────────────────────────────
#  SLIC Superpixel aggregator
# ─────────────────────────────────────────────────────────────

class SuperpixelAggregator:
    
    def __init__(self, cfg: Config):
        self.cfg  = cfg
        self._slic = cv2.ximgproc.createSLICSuperpixelSLIC if hasattr(
            cv2, 'ximgproc'
        ) else None
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            import cv2
            test = np.zeros((64, 64, 3), dtype=np.uint8)
            slic = cv2.ximgproc.createSuperpixelSLIC(test, region_size=16)
            return True
        except Exception:
            log.warning("cv2.ximgproc not available. Falling back to pixel-level GMM.")
            return False

    def compute_labels(
        self, img_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int]:
        if not self._available:
            return None, 0
        cfg = self.cfg
        slic = cv2.ximgproc.createSuperpixelSLIC(
            img_bgr,
            algorithm=cv2.ximgproc.SLIC,
            region_size=cfg.slic_region_size,
            ruler=cfg.slic_ruler,
        )
        slic.iterate(10)
        labels = slic.getLabels()   # (H, W) int32
        n      = slic.getNumberOfSuperpixels()
        return labels, n

    def aggregate_features(
        self,
        feat_map: np.ndarray,
        labels: np.ndarray,
        n_labels: int,
    ) -> np.ndarray:
        d = feat_map.shape[2]
        flat_feat   = feat_map.reshape(-1, d)
        flat_labels = labels.reshape(-1)

        # Vectorised mean per label using np.bincount
        sp_features = np.zeros((n_labels, d), dtype=np.float32)
        counts      = np.bincount(flat_labels, minlength=n_labels).astype(np.float32)
        for dim in range(d):
            sp_features[:, dim] = (
                np.bincount(flat_labels, weights=flat_feat[:, dim], minlength=n_labels)
                / np.maximum(counts, 1)
            )
        return sp_features

    def paint_scores(
        self,
        sp_scores: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        return sp_scores[labels].astype(np.float32)

# ─────────────────────────────────────────────────────────────
#  Optical flow temporal propagator
# ─────────────────────────────────────────────────────────────

class FlowPropagator:
    """
    Warps previous binary mask using Farneback optical flow.
    Provides much better temporal consistency than naive EMA.
    """

    def __init__(self):
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_mask: Optional[np.ndarray] = None

    def update(
        self,
        gray: np.ndarray,
        curr_mask: np.ndarray,
        blend_alpha: float,
    ) -> np.ndarray:
        if self._prev_gray is None or self._prev_mask is None:
            self._prev_gray = gray.copy()
            self._prev_mask = curr_mask.copy()
            return curr_mask

        # Compute dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None,
            pyr_scale=0.5, levels=2, winsize=13,
            iterations=2, poly_n=5, poly_sigma=1.1,
            flags=0
        )

        # Warp previous mask
        h, w = gray.shape
        map_x = (np.arange(w, dtype=np.float32).reshape(1, w)
                 + flow[:, :, 0])
        map_y = (np.arange(h, dtype=np.float32).reshape(h, 1)
                 + flow[:, :, 1])

        warped = cv2.remap(
            self._prev_mask.astype(np.float32),
            map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Blend warped previous + current
        blended = (
            blend_alpha       * curr_mask.astype(np.float32) +
            (1.0 - blend_alpha) * warped
        )
        result = (blended > 127).astype(np.uint8) * 255

        self._prev_gray = gray.copy()
        self._prev_mask = result.copy()
        return result

    def reset(self):
        self._prev_gray = None
        self._prev_mask = None

# ─────────────────────────────────────────────────────────────
#  GMM model
# ─────────────────────────────────────────────────────────────

class TraversabilityGMM:
    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self.gmm       = None
        self.scaler    = None
        self.threshold = None
        self.fitted    = False

    def _best_k(self, x: np.ndarray) -> int:
        if not self.cfg.use_bic:
            return self.cfg.gmm_k
        best_bic, best_k = np.inf, self.cfg.gmm_k_range[0]
        for k in range(self.cfg.gmm_k_range[0],
                       min(self.cfg.gmm_k_range[1], x.shape[0] - 1) + 1):
            try:
                g = GaussianMixture(
                    n_components=k,
                    covariance_type=self.cfg.gmm_covariance,
                    reg_covar=self.cfg.gmm_reg_covar,
                    max_iter=50, n_init=1,
                    random_state=self.cfg.random_seed,
                )
                g.fit(x)
                b = g.bic(x)
                if b < best_bic:
                    best_bic, best_k = b, k
            except Exception:
                pass
        return best_k

    def fit(self, features: np.ndarray) -> bool:
        if len(features) < self.cfg.min_seed_pixels:
            return False

        # Subsample
        n = min(len(features), self.cfg.max_fit_samples)
        rng = np.random.default_rng(self.cfg.random_seed)
        idx = rng.choice(len(features), n, replace=False)
        X   = features[idx]

        scaler   = StandardScaler()
        x_scaled = scaler.fit_transform(X)

        k = self._best_k(x_scaled)
        log.info("  GMM fit k=%d on %d samples", k, n)

        gmm = GaussianMixture(
            n_components=k,
            covariance_type=self.cfg.gmm_covariance,
            reg_covar=self.cfg.gmm_reg_covar,
            max_iter=self.cfg.gmm_max_iter,
            n_init=self.cfg.gmm_n_init,
            random_state=self.cfg.random_seed,
            warm_start=False,
        )
        gmm.fit(x_scaled)

        seed_scores = gmm.score_samples(x_scaled)

        if self.cfg.use_otsu_threshold:
            # Otsu on score histogram (separates traversable from non)
            norm_scores = cv2.normalize(
                seed_scores.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            otsu_val, _ = cv2.threshold(
                norm_scores, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # Map Otsu val back to score space
            s_min = seed_scores.min()
            s_max = seed_scores.max()
            self.threshold = s_min + (otsu_val / 255.0) * (s_max - s_min)
            self.threshold -= self.cfg.threshold_margin
        else:
            self.threshold = (
                np.percentile(seed_scores, self.cfg.threshold_percentile)
                - self.cfg.threshold_margin
            )

        self.gmm    = gmm
        self.scaler = scaler
        self.fitted = True
        log.info("  Threshold=%.3f", self.threshold)
        return True

    def score(self, features: np.ndarray, chunk: int = 50_000) -> np.ndarray:
        assert self.fitted
        out = np.empty(len(features), dtype=np.float32)
        for s in range(0, len(features), chunk):
            e = min(s + chunk, len(features))
            out[s:e] = self.gmm.score_samples(
                self.scaler.transform(features[s:e])
            ).astype(np.float32)
        return out

    def score_map(
        self,
        feat_map: np.ndarray,
        roi_mask: np.ndarray,
        sp_aggregator: SuperpixelAggregator,
        img_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Returns float32 score map (H, W).
        Uses superpixels when available for large speedup.
        """
        assert self.fitted
        h, w, d = feat_map.shape
        fill_val = self.threshold - 10.0

        if sp_aggregator._available and self.cfg.use_superpixels:
            # ── superpixel path ──────────────────────────────
            labels, n_sp = sp_aggregator.compute_labels(img_bgr)
            if labels is not None and n_sp > 0:
                sp_feat = sp_aggregator.aggregate_features(feat_map, labels, n_sp)
                sp_scores = self.score(sp_feat)
                pixel_scores = sp_aggregator.paint_scores(sp_scores, labels)
                # Zero out non-ROI
                pixel_scores[roi_mask == 0] = fill_val
                return pixel_scores

        # ── pixel-level fallback ─────────────────────────────
        flat   = feat_map.reshape(-1, d)
        out    = np.full(h * w, fill_val, dtype=np.float32)
        roi_ix = np.where(roi_mask.reshape(-1) > 0)[0]
        out[roi_ix] = self.score(flat[roi_ix])
        return out.reshape(h, w)

# ─────────────────────────────────────────────────────────────
#  Seed mask & ROI (cached per resolution)
# ─────────────────────────────────────────────────────────────

class MaskCache:
    def __init__(self, cfg: Config):
        self.cfg  = cfg
        self._seed: Optional[np.ndarray] = None
        self._roi:  Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int, int]] = None

    def get(self, h: int, w: int):
        if self._shape == (h, w):
            return self._seed, self._roi
        cfg = self.cfg

        # ── Trapezoidal seed ──────────────────────────────
        seed = np.zeros((h, w), dtype=np.uint8)
        y_top = int(h * cfg.seed_top_frac)
        y_bot = int(h * cfg.seed_bot_frac)
        for y in range(y_top, y_bot):
            frac   = (y - y_top) / max(y_bot - y_top, 1)
            half_w = int(w * cfg.seed_cx_frac + w * 0.18 * frac)
            cx     = w // 2
            seed[y, max(0, cx - half_w): min(w, cx + half_w)] = 255

        # ── ROI ───────────────────────────────────────────
        roi = np.zeros((h, w), dtype=np.uint8)
        roi[int(h * cfg.roi_top_frac):, :] = 255

        self._seed  = seed
        self._roi   = roi
        self._shape = (h, w)
        return seed, roi

# ─────────────────────────────────────────────────────────────
#  Online sample buffer
# ─────────────────────────────────────────────────────────────

class SeedSampler:
    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self._buffer: Deque[np.ndarray] = deque(maxlen=cfg.adapt_history_frames)

    def add(
        self,
        feat_map: np.ndarray,
        seed_mask: np.ndarray,
        prev_traversable: Optional[np.ndarray],
    ) -> None:
        if prev_traversable is not None:
            # Only sample pixels that were PREVIOUSLY confirmed traversable
            # → avoids drifting into non-traversable classes over time
            combined = cv2.bitwise_and(seed_mask, prev_traversable)
            # Require sufficient overlap
            if cv2.countNonZero(combined) < self.cfg.min_seed_pixels:
                combined = seed_mask
        else:
            combined = seed_mask

        feats = flat_features(feat_map, combined)
        if len(feats) >= self.cfg.min_seed_pixels:
            self._buffer.append(feats)

    def get(self) -> Optional[np.ndarray]:
        if not self._buffer:
            return None
        return np.vstack(list(self._buffer))

# ─────────────────────────────────────────────────────────────
#  Post-processing
# ─────────────────────────────────────────────────────────────

def postprocess(
    mask: np.ndarray,
    roi_mask: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    mask = cv2.bitwise_and(mask, roi_mask)

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,  k, iterations=cfg.morph_open_iter
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, k, iterations=cfg.morph_close_iter
    )

    # Fill holes
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)

    # Filter components
    n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(
        filled, connectivity=8
    )
    if n_lab <= 1:
        return np.zeros_like(mask)

    h, w   = mask.shape
    cx_img = w / 2.0
    scored = []
    for lbl in range(1, n_lab):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < cfg.min_region_area:
            continue
        cy = centroids[lbl][1]
        cx = centroids[lbl][0]
        s  = area + 0.35 * cy * area / (h * w) - 0.12 * abs(cx - cx_img)
        scored.append((s, lbl))

    scored.sort(reverse=True)
    out = np.zeros_like(mask)
    for _, lbl in scored[: cfg.top_n_regions]:
        out[labels == lbl] = 255
    return out

# ─────────────────────────────────────────────────────────────
#  Path tracer
# ─────────────────────────────────────────────────────────────

def trace_path(traversable: np.ndarray, cfg: Config) -> List[Tuple[int, int]]:
    ys = np.where(traversable > 0)[0]
    if len(ys) == 0:
        return []

    h, w  = traversable.shape
    dist  = cv2.distanceTransform(traversable, cv2.DIST_L2, 3)
    bias  = np.abs(np.arange(w, dtype=np.float32) - w / 2) * 0.04

    y_bot = int(np.max(ys))
    y_top = int(np.min(ys))

    raw: List[Tuple[int, int]] = []
    for y in range(y_bot, y_top, -cfg.path_step):
        row = dist[y]
        if row.max() == 0:
            continue
        bx = int(np.argmax(row - bias))
        if traversable[y, bx] > 0:
            raw.append((bx, y))

    if len(raw) < cfg.min_path_points:
        return []

    win = cfg.path_smooth_window
    return [
        (int(np.mean([p[0] for p in raw[max(0, i - win): i + win + 1]])), py)
        for i, (_, py) in enumerate(raw)
    ]

# ─────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────

# Precomputed colour gradient for path
_PATH_COLORS = np.array([
    [int(255 * t), int(255 * (1 - t * 0.5)), 255]
    for t in np.linspace(0, 1, 256)
], dtype=np.uint8)

def draw_path(img: np.ndarray, path: List[Tuple[int, int]]) -> None:
    if len(path) < 2:
        return
    n = len(path)
    for i in range(n - 1):
        t     = i / n
        ci    = int(t * 255)
        color = tuple(int(c) for c in _PATH_COLORS[ci])
        thick = max(2, int(8 * (1 - t * 0.6)))
        cv2.line(img, path[i], path[i + 1], color, thick, cv2.LINE_AA)
    cv2.circle(img, path[0], 7, (0, 255, 255), -1, cv2.LINE_AA)

def draw_overlay(
    frame: np.ndarray,
    traversable: np.ndarray,
    path: List[Tuple[int, int]],
    seed_mask: Optional[np.ndarray],
    fps: float,
    frame_idx: int,
    gmm_updated: bool,
    cfg: Config,
    score_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    result = frame.copy()
    h, w   = result.shape[:2]

    # Green overlay
    green_layer = np.zeros_like(result)
    green_layer[traversable > 0] = (0, 200, 0)
    cv2.addWeighted(green_layer, 0.38, result, 0.62, 0, result)

    # Contour
    cnts, _ = cv2.findContours(
        traversable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, cnts, -1, (0, 255, 0), 2)

    # Seed
    if cfg.show_seed_region and seed_mask is not None:
        sc, _ = cv2.findContours(
            seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, sc, -1, (0, 180, 255), 1)

    # Path
    draw_path(result, path)

    # Horizon
    cv2.line(
        result, (0, int(h * cfg.horizon_frac)),
        (w, int(h * cfg.horizon_frac)), (255, 80, 0), 1
    )

    # HUD
    cov  = 100.0 * np.count_nonzero(traversable) / traversable.size
    hud  = [
        f"Frame : {frame_idx:5d}",
        f"FPS   : {fps:5.1f}",
        f"Cover : {cov:4.1f}%",
        f"Path  : {len(path)} pts",
        f"GMM   : {'UPDATE' if gmm_updated else 'cache'}",
    ]
    ph_px = len(hud) * 22 + 10
    result[:ph_px, :205] = (result[:ph_px, :205].astype(np.float32) * 0.3).astype(np.uint8)
    for i, txt in enumerate(hud):
        clr = (0, 255, 255) if (i == 4 and gmm_updated) else (220, 220, 220)
        cv2.putText(result, txt, (8, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, clr, 1, cv2.LINE_AA)

    # Score heatmap side panel
    if cfg.show_score_map and score_map is not None:
        norm = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat = cv2.resize(heat, (w, h))
        result = np.hstack([result, heat])

    return result

# ─────────────────────────────────────────────────────────────
#  Click seeder
# ─────────────────────────────────────────────────────────────

class ClickSeeder:
    def __init__(self):
        self.points: List[Tuple[int, int]] = []
        self.dirty  = False

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.dirty = True
            log.info("Seed added (%d,%d)", x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.clear()
            self.dirty = True
            log.info("Seeds cleared")

    def mask(self, h: int, w: int, r: int = 28) -> Optional[np.ndarray]:
        if not self.points:
            return None
        m = np.zeros((h, w), dtype=np.uint8)
        for px, py in self.points:
            cv2.circle(m, (px, py), r, 255, -1)
        return m

    def pop_dirty(self) -> bool:
        d, self.dirty = self.dirty, False
        return d

# ─────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.gmm    = TraversabilityGMM(cfg)
        self.sampler = SeedSampler(cfg)
        self.sp     = SuperpixelAggregator(cfg)
        self.flow   = FlowPropagator()
        self.prep   = Preprocessor(cfg)
        self.masks  = MaskCache(cfg)
        self.clicker = ClickSeeder()

        self._spatial = _SPATIAL_CACHE
        self._prev_traversable: Optional[np.ndarray] = None
        self._score_ema: Optional[np.ndarray]        = None
        self._fps_buf: Deque[float] = deque(maxlen=40)
        self._t0      = time.perf_counter()
        self.frame_idx = 0
        self.gmm_updated = False

    def _should_fit(self) -> bool:
        fi = self.frame_idx
        return (fi == 0
                or fi % self.cfg.adapt_every_n_frames == 0
                or self.clicker.pop_dirty())

    def _fit_gmm(self, feat_map: np.ndarray, h: int, w: int) -> None:
        seed_mask, _ = self.masks.get(h, w)
        eff_seed = self.clicker.mask(h, w) or seed_mask
        self.sampler.add(feat_map, eff_seed, self._prev_traversable)
        samples = self.sampler.get()
        if samples is not None:
            self.gmm_updated = self.gmm.fit(samples)
        else:
            self.gmm_updated = False

    def process(self, raw_frame: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        h, w = raw_frame.shape[:2]

        # ── Pre-process (CLAHE) ───────────────────────────
        frame = self.prep(raw_frame)

        # ── Features ─────────────────────────────────────
        feat_map = extract_features(frame, self._spatial)

        # ── Seed & ROI masks ──────────────────────────────
        seed_mask, roi_mask = self.masks.get(h, w)

        # ── GMM fit ───────────────────────────────────────
        if self._should_fit():
            self._fit_gmm(feat_map, h, w)
        else:
            self.gmm_updated = False

        # ── Score map ────────────────────────────────────
        if self.gmm.fitted:
            raw_score = self.gmm.score_map(
                feat_map, roi_mask, self.sp, frame
            )

            # Score EMA
            if self._score_ema is None or self._score_ema.shape != raw_score.shape:
                self._score_ema = raw_score.copy()
            else:
                a = cfg.score_ema_alpha
                self._score_ema = a * raw_score + (1.0 - a) * self._score_ema

            # Threshold
            binary = (self._score_ema >= self.gmm.threshold
                      ).astype(np.uint8) * 255

            # Post-process
            clean = postprocess(binary, roi_mask, cfg)

            # ── Temporal consistency ─────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cfg.use_optical_flow:
                traversable = self.flow.update(
                    gray, clean, cfg.flow_blend_alpha
                )
            else:
                traversable = clean

        else:
            traversable  = seed_mask.copy()
            raw_score    = None

        self._prev_traversable = traversable

        # ── Path ─────────────────────────────────────────
        path = trace_path(traversable, cfg)

        # ── FPS ──────────────────────────────────────────
        t_now = time.perf_counter()
        self._fps_buf.append(1.0 / max(t_now - self._t0, 1e-6))
        self._t0 = t_now
        fps = float(np.mean(self._fps_buf))

        # ── Draw ─────────────────────────────────────────
        vis_seed = self.clicker.mask(h, w) or seed_mask
        result   = draw_overlay(
            raw_frame, traversable, path, vis_seed,
            fps, self.frame_idx, self.gmm_updated, cfg,
            score_map=raw_score,
        )

        self.frame_idx += 1
        return result

# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fast GMM Traversability")
    parser.add_argument("video",      nargs="?", default=None)
    parser.add_argument("--score-map", action="store_true")
    parser.add_argument("--no-flow",   action="store_true",
                        help="Disable optical flow temporal propagation")
    parser.add_argument("--bic",       action="store_true",
                        help="Use BIC component selection (slower, more accurate)")
    parser.add_argument("--full-cov",  action="store_true",
                        help="Use full covariance GMM (slower)")
    parser.add_argument("--width",     type=int, default=None,
                        help="Processing width override")
    args = parser.parse_args()

    cfg = Config()
    if args.video:      cfg.video_path    = args.video
    if args.score_map:  cfg.show_score_map = True
    if args.no_flow:    cfg.use_optical_flow = False
    if args.bic:        cfg.use_bic = True
    if args.full_cov:   cfg.gmm_covariance = "full"
    if args.width:      cfg.max_proc_width = args.width

    log.info("Video: %s", cfg.video_path)
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {cfg.video_path}")

    w_orig  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = min(1.0, cfg.max_proc_width / float(w_orig))
    pw, ph = int(w_orig * scale), int(h_orig * scale)

    log.info("Source %dx%d @%.1ffps | Processing %dx%d", w_orig, h_orig, fps_src, pw, ph)
    log.info("Superpixels: %s | Optical flow: %s | BIC: %s",
             cfg.use_superpixels, cfg.use_optical_flow, cfg.use_bic)

    pipeline = Pipeline(cfg)

    WIN = "GMM Traversability"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, pw, ph)
    cv2.setMouseCallback(WIN, pipeline.clicker.callback)

    log.info("Controls: ESC/Q=quit | L-click=add seed | R-click=clear")

    t_wall = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        proc = (cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
                if scale < 1.0 else frame.copy())

        result = pipeline.process(proc)
        cv2.imshow(WIN, result)

        if pipeline.frame_idx % 60 == 0:
            avg = pipeline.frame_idx / max(time.time() - t_wall, 1e-6)
            log.info("Frame %d/%d | avg %.1f fps", pipeline.frame_idx, n_total, avg)

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t_wall
    log.info("Done: %d frames / %.1fs = %.1f fps",
             pipeline.frame_idx, elapsed,
             pipeline.frame_idx / max(elapsed, 1e-6))

if __name__ == "__main__":
    main()