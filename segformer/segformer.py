#!/usr/bin/env python3

import time
import cv2
import numpy as np
import torch
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

VIDEO_PATH = "vid.mp4"

MAX_PROC_WIDTH    = 480
SEG_IMGSZ         = 256
KEYFRAME_INTERVAL = 4

TRAVERSABLE_IDS = {
    3, 6, 9, 11, 13, 29, 46, 52, 53, 54, 61, 91, 94, 121,
}

OBSTACLE_IDS = {
    0, 1, 4, 12, 16, 17, 20, 21, 26, 32, 34, 38, 60, 68,
    72, 76, 80, 83, 93, 102, 113, 114, 116, 122, 126, 127, 128,
}

ROI_TOP_FRAC    = 0.35
MORPH_KERNEL    = 7
MIN_REGION_AREA = 500
TOP_N_REGIONS   = 3

# --- smoothing knobs -------------------------------------------------------
EMA_ALPHA       = 0.28     # lower = smoother in time (0..1)
HYST_HI         = 0.55     # enter "traversable"
HYST_LO         = 0.40     # leave "traversable"
EDGE_BLUR       = 15       # odd; spatial boundary softening (px)
CONTOUR_SMOOTH  = 9        # moving-average window on contour points
CONTOUR_EPS     = 1.2      # approxPolyDP epsilon (px)
# ---------------------------------------------------------------------------


class Models:

    def __init__(self):
        self.seg_proc  = None
        self.seg_model = None
        self.device    = torch.device("cpu")
        self._loaded   = False

    def load(self):
        if self._loaded:
            return
        print("  Loading SegFormer-B0 (ADE20K) …")
        self.seg_proc = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        ).to(self.device)
        self.seg_model.eval()

        print("  Warming up …")
        dummy = np.zeros((SEG_IMGSZ, SEG_IMGSZ, 3), dtype=np.uint8)
        self._run_segformer(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))

        self._loaded = True
        print("  Models ready.\n")

    @torch.inference_mode()
    def _run_segformer(self, img_rgb):
        inputs = self.seg_proc(images=img_rgb, return_tensors="pt").to(self.device)
        return self.seg_model(**inputs).logits

    def segment(self, frame_bgr, target_h, target_w):
        small = cv2.resize(frame_bgr, (SEG_IMGSZ, SEG_IMGSZ),
                           interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        logits = self._run_segformer(rgb)
        upsampled = torch.nn.functional.interpolate(
            logits, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )
        seg_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

        trav = np.isin(seg_map, list(TRAVERSABLE_IDS)).astype(np.uint8) * 255
        obs  = np.isin(seg_map, list(OBSTACLE_IDS)).astype(np.uint8) * 255
        return trav, obs


MODELS = Models()


class FlowTracker:

    def __init__(self):
        self._flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self._prev_gray = None
        self._cached_flow = None
        self._acc_flow = None            # accumulated flow since last keyframe
        self._grid_x = None
        self._grid_y = None
        self._grid_shape = None

    def _ensure_grid(self, h, w):
        if self._grid_shape != (h, w):
            self._grid_x = np.arange(w, dtype=np.float32).reshape(1, w)
            self._grid_y = np.arange(h, dtype=np.float32).reshape(h, 1)
            self._grid_shape = (h, w)

    def update_flow(self, curr_gray):
        if self._prev_gray is None:
            self._prev_gray = curr_gray.copy()
            self._cached_flow = None
            return
        self._cached_flow = self._flow.calc(self._prev_gray, curr_gray, None)
        self._prev_gray = curr_gray.copy()

        if self._acc_flow is None or self._acc_flow.shape != self._cached_flow.shape:
            self._acc_flow = self._cached_flow.copy()
        else:
            self._acc_flow += self._cached_flow

    def reset_accumulation(self):
        self._acc_flow = None

    def warp(self, mask_f32):
        """Warp a keyframe mask by the accumulated flow since the keyframe."""
        if self._acc_flow is None:
            return mask_f32.copy()

        h, w = mask_f32.shape[:2]
        self._ensure_grid(h, w)
        map_x = self._grid_x + self._acc_flow[:, :, 0]
        map_y = self._grid_y + self._acc_flow[:, :, 1]

        return cv2.remap(mask_f32, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def reset(self):
        self._prev_gray = None
        self._cached_flow = None
        self._acc_flow = None
        self._grid_shape = None


def apply_roi(mask, h):
    mask[:int(h * ROI_TOP_FRAC), :] = 0
    return mask


def remove_obstacles(trav_mask, obs_mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    obs_dilated = cv2.dilate(obs_mask, k, iterations=1)
    trav_mask[obs_dilated > 0] = 0
    return trav_mask


def morph_cleanup(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def keep_best_components(mask, h, w):
    n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n_lab <= 1:
        return np.zeros_like(mask)

    cx_img = w / 2.0
    scored = []
    for lbl in range(1, n_lab):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < MIN_REGION_AREA:
            continue
        cx, cy = centroids[lbl]
        s = area + 0.3 * cy * area / (h * w) - 0.08 * abs(cx - cx_img)
        scored.append((s, lbl))

    scored.sort(reverse=True)
    out = np.zeros_like(mask)
    for _, lbl in scored[:TOP_N_REGIONS]:
        out[labels == lbl] = 255
    return out


class TemporalSmoother:
    """EMA on a soft mask + hysteresis thresholding -> flicker-free binary mask."""

    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self._soft = None
        self._bin  = None

    def update(self, curr_mask):
        curr = curr_mask.astype(np.float32) / 255.0

        if self._soft is None or self._soft.shape != curr.shape:
            self._soft = curr.copy()
            self._bin  = (curr > 0.5)
        else:
            self._soft = self.alpha * curr + (1.0 - self.alpha) * self._soft

        # hysteresis: keep previous state in the ambiguous band
        on  = self._soft >= HYST_HI
        off = self._soft <= HYST_LO
        self._bin = np.where(on, True, np.where(off, False, self._bin))

        binary = self._bin.astype(np.uint8) * 255
        return binary, self._soft.copy()

    def reset(self):
        self._soft = None
        self._bin = None


def spatial_smooth(mask):
    """Soften the boundary so contours are round instead of blocky."""
    k = EDGE_BLUR | 1
    blur = cv2.GaussianBlur(mask, (k, k), 0)
    return (blur > 127).astype(np.uint8) * 255


def smooth_contour(cnt, win=CONTOUR_SMOOTH):
    """Circular moving-average smoothing of a closed contour."""
    pts = cnt.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    if n < win * 2:
        return cnt
    pad = np.vstack([pts[-win:], pts, pts[:win]])
    kernel = np.ones(2 * win + 1, dtype=np.float32) / (2 * win + 1)
    xs = np.convolve(pad[:, 0], kernel, mode="same")[win:win + n]
    ys = np.convolve(pad[:, 1], kernel, mode="same")[win:win + n]
    return np.stack([xs, ys], axis=1).round().astype(np.int32).reshape(-1, 1, 2)


def get_smooth_contours(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_REGION_AREA:
            continue
        c = smooth_contour(c)
        c = cv2.approxPolyDP(c, CONTOUR_EPS, True)
        out.append(c)
    return out


def draw_overlay(frame, mask, soft, contours, fps, frame_idx, is_keyframe):
    h, w = frame.shape[:2]
    result = frame.copy()

    # feathered green fill driven by the soft (blurred) mask
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (EDGE_BLUR | 1,) * 2, 0)
    a = (a * 0.38)[..., None]
    green = np.zeros_like(result, dtype=np.float32)
    green[:, :] = (0, 200, 0)
    result = (result.astype(np.float32) * (1.0 - a) + green * a).astype(np.uint8)

    # smooth traversable contour
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

    y_roi = int(h * ROI_TOP_FRAC)
    cv2.line(result, (0, y_roi), (w, y_roi), (255, 100, 0), 1, cv2.LINE_AA)

    cov = 100.0 * np.count_nonzero(mask) / mask.size
    hud = [
        f"Frame : {frame_idx}",
        f"FPS   : {fps:.1f}",
        f"Cover : {cov:.1f}%",
        f"Mode  : {'KEYFRAME' if is_keyframe else 'Flow Track'}",
    ]
    ph = len(hud) * 22 + 10
    result[:ph, :230] = (result[:ph, :230].astype(np.float32) * 0.3).astype(np.uint8)
    for i, txt in enumerate(hud):
        clr = (0, 255, 255) if (is_keyframe and i == 3) else (220, 220, 220)
        cv2.putText(result, txt, (8, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, clr, 1, cv2.LINE_AA)

    return result


def main():
    print("=" * 64)
    print("  SegFormer-B0 Traversable Area Detection")
    print("  Live View Only  ·  CPU  ·  DIS Flow Tracking  ·  Smooth Contours")
    print("=" * 64)

    MODELS.load()

    print(f"  Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = min(1.0, MAX_PROC_WIDTH / float(w_orig))
    pw, ph = int(w_orig * scale), int(h_orig * scale)

    print(f"  Source : {w_orig}×{h_orig} @ {fps_src:.1f} fps ({total} frames)")
    print(f"  Process: {pw}×{ph}")
    print(f"  Keyframe interval: every {KEYFRAME_INTERVAL} frames")
    print(f"\n  Press ESC or 'q' to quit.\n")

    tracker  = FlowTracker()
    smoother = TemporalSmoother()

    key_trav = None      # raw keyframe masks (float32 0..1) - never fed back
    key_obs  = None

    frame_idx = 0
    t_start   = time.time()
    t_prev    = time.time()
    fps_disp  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_now = time.time()
        inst = 1.0 / max(t_now - t_prev, 1e-6)
        fps_disp = inst if frame_idx == 0 else 0.9 * fps_disp + 0.1 * inst
        t_prev = t_now

        proc = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_LINEAR) \
            if scale < 1.0 else frame.copy()

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        tracker.update_flow(gray)

        is_keyframe = (frame_idx % KEYFRAME_INTERVAL == 0) or key_trav is None

        if is_keyframe:
            trav_raw, obs_raw = MODELS.segment(proc, ph, pw)
            key_trav = trav_raw.astype(np.float32)
            key_obs  = obs_raw.astype(np.float32)
            tracker.reset_accumulation()
            trav = trav_raw
            obs  = obs_raw
        else:
            trav = (tracker.warp(key_trav) > 127).astype(np.uint8) * 255
            obs  = (tracker.warp(key_obs)  > 127).astype(np.uint8) * 255

        trav = apply_roi(trav, ph)
        trav = remove_obstacles(trav, obs)
        trav = morph_cleanup(trav)
        trav = keep_best_components(trav, ph, pw)

        trav, soft = smoother.update(trav)
        trav = spatial_smooth(trav)

        contours = get_smooth_contours(trav)

        vis = draw_overlay(proc, trav, soft, contours,
                           fps_disp, frame_idx, is_keyframe)
        cv2.imshow("SegFormer Traversable Area (Live)", vis)

        frame_idx += 1
        if frame_idx % 30 == 0:
            avg = frame_idx / max(time.time() - t_start, 1e-6)
            print(f"  Frame {frame_idx}/{total} | avg {avg:.1f} fps")

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            print("  Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n  Done: {frame_idx} frames in {elapsed:.1f}s "
          f"= {frame_idx / max(elapsed, 1e-6):.1f} fps avg")
    print("  No output saved (live view only).")


if __name__ == "__main__":
    main()