import cv2
import numpy as np


class ClassicalPipeline:
    def __init__(self, process_width=640):
        self.prev_score = None
        self.prev_traversable = None
        self.alpha = 0.7
        self.process_width = process_width
        self.ground_mean = None
        self.ground_std = None

    def _score_component(self, mask, frame_h, frame_w):
        """Score a connected component with no hard geometric assumptions."""
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) == 0:
            return -1.0

        area = len(y_coords)
        area_ratio = area / (frame_h * frame_w)

        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        height_ratio = (y_max - y_min + 1) / frame_h

        # Soft centeredness: mild preference, not requirement
        center_x = frame_w // 2
        comp_center_x = int(np.mean(x_coords))
        center_offset = abs(comp_center_x - center_x) / (frame_w / 2)
        center_factor = 1.0 + 0.2 * (1.0 - center_offset)

        score = area_ratio * height_ratio * center_factor

        # Flexible bottom: bonus if component exists within bottom 10%
        # (soft preference, not a hard gate)
        bottom_zone = int(frame_h * 0.90)
        if np.any(mask[bottom_zone:, :]):
            score *= 2.0

        return score

    def _select_best_component(self, floor_mask, h, w):
        """Pick the best traversable component (flexible, no hard bottom-touch)."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(floor_mask, 4)
        result = np.zeros_like(floor_mask)

        candidates = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 300:
                continue
            component_mask = (labels == label_id).astype(np.uint8) * 255
            sc = self._score_component(component_mask, h, w)
            if sc > 0:
                candidates.append((label_id, sc))

        if candidates:
            best = max(candidates, key=lambda x: x[1])[0]
            result[labels == best] = 255
        elif num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            result[labels == largest] = 255

        return result

    def _build_height_weight(self, h, w):
        
        weight = np.zeros((h, w), dtype=np.float32)
        full_start = int(h * 0.80)
        fade_start = int(h * 0.65)

        weight[full_start:, :] = 1.0
        for y in range(fade_start, full_start):
            weight[y, :] = (y - fade_start) / max(full_start - fade_start, 1)
        return weight

    def process(self, image, yolo_traversable=None, yolo_obstacle=None, return_original_res=True):
        orig_h, orig_w = image.shape[:2]
        scale = self.process_width / orig_w
        if scale < 1.0:
            new_w = self.process_width
            new_h = int(orig_h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_w, new_h = orig_w, orig_h

        h, w = image.shape[:2]

        # ── Color spaces ──
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # ── Soft height gradient (replaces hard ROI cutoff) ──
        height_weight = self._build_height_weight(h, w)

        # ── YOLO-validated ground color sampling ──
        ground_sampled = False
        yolo_resized = None
        if yolo_traversable is not None:
            yolo_resized = cv2.resize(yolo_traversable, (w, h), interpolation=cv2.INTER_NEAREST)
            # Only sample ground color from bottom pixels CONFIRMED by YOLO
            ground_zone = np.zeros((h, w), dtype=np.uint8)
            ground_zone[int(h * 0.85):h - 3, :] = 255
            valid_ground = cv2.bitwise_and(ground_zone, yolo_resized)
            valid_pixels = lab[valid_ground > 0]
            if len(valid_pixels) > 50:
                ground_pixels = valid_pixels.astype(np.float32)
                ground_sampled = True

        if not ground_sampled:
            # Fallback: conservative center-bottom strip
            sy1, sy2 = int(h * 0.90), h - 3
            sx1, sx2 = int(w * 0.30), int(w * 0.70)
            ground_pixels = lab[sy1:sy2, sx1:sx2].reshape(-1, 3).astype(np.float32)

        # Update temporal ground model
        if len(ground_pixels) > 20:
            curr_mean = np.mean(ground_pixels, axis=0)
            curr_std = np.std(ground_pixels, axis=0) + 5.0
            if self.ground_mean is not None:
                self.ground_mean = 0.85 * self.ground_mean + 0.15 * curr_mean
                self.ground_std = 0.85 * self.ground_std + 0.15 * curr_std
            else:
                self.ground_mean = curr_mean
                self.ground_std = curr_std

        if self.ground_mean is None:
            self.ground_mean = np.array([128.0, 128.0, 128.0], dtype=np.float32)
            self.ground_std = np.array([25.0, 25.0, 25.0], dtype=np.float32)

        # ── Color distance ──
        diff = np.abs(lab.astype(np.float32) - self.ground_mean)
        norm_dist = diff / self.ground_std
        mean_norm_dist = np.mean(norm_dist, axis=2)
        color_score = np.clip(1.0 - mean_norm_dist / 2.0, 0, 1)

        # ── Edge detection ──
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray_blur, 40, 120)
        edge_barrier = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        edge_penalty = cv2.GaussianBlur(edge_barrier.astype(np.float32), (9, 9), 0)
        edge_penalty = cv2.normalize(edge_penalty, None, 0, 1, cv2.NORM_MINMAX)

        # ── Texture ──
        blur_img = cv2.GaussianBlur(gray, (7, 7), 0)
        texture = cv2.absdiff(gray, blur_img).astype(np.float32)
        texture = cv2.blur(texture, (11, 11))
        texture = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)

        # ── Saturation (reduced weight — not all roads are low-sat) ──
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        sat_smooth = cv2.GaussianBlur(sat, (9, 9), 0)

        # ── Classical score (surface-agnostic weighting) ──
        classical_score = (
            0.50 * color_score * (1.0 - 0.5 * edge_penalty)
            + 0.20 * (1.0 - edge_penalty)
            + 0.15 * (1.0 - texture)
            + 0.10 * (1.0 - sat_smooth)
            + 0.05   # base floor so nothing is pure zero
        )
        classical_score = cv2.GaussianBlur(classical_score, (9, 9), 0)

        # ── Temporal smoothing (on raw score, before height weighting) ──
        if self.prev_score is not None:
            prev_resized = cv2.resize(self.prev_score, (w, h), interpolation=cv2.INTER_LINEAR)
            classical_score = self.alpha * classical_score + (1 - self.alpha) * prev_resized
        self.prev_score = classical_score.copy()

        # Apply soft height gradient AFTER temporal smoothing
        classical_weighted = classical_score * height_weight

        # ── Edge carving mask ──
        strong_edges = cv2.Canny(gray_blur, 60, 160)
        edge_carve = cv2.dilate(strong_edges, np.ones((3, 3), np.uint8), iterations=1)

        # Morphological kernels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # ══════════════════════════════════════
        # ══ FUSION: YOLO semantic + Classical ══
        # ══════════════════════════════════════
        has_yolo = (yolo_traversable is not None and np.any(yolo_traversable))

        if has_yolo:
            # Soft YOLO map with height weighting
            yolo_weighted = (yolo_resized.astype(np.float32) / 255.0) * height_weight
            yolo_soft = cv2.GaussianBlur(yolo_weighted, (15, 15), 0)

            # Weighted blend
            combined = 0.55 * yolo_soft + 0.45 * classical_weighted
            floor_mask = (combined > 0.35).astype(np.uint8) * 255

            # Edge carving
            floor_mask = cv2.bitwise_and(floor_mask, cv2.bitwise_not(edge_carve))

            # Obstacle removal
            if yolo_obstacle is not None:
                obs = cv2.resize(yolo_obstacle, (w, h), interpolation=cv2.INTER_NEAREST)
                obs = cv2.dilate(obs, np.ones((13, 13), np.uint8), iterations=1)
                floor_mask = cv2.bitwise_and(floor_mask, cv2.bitwise_not(obs))

            # Morphological cleanup
            floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

            traversable = self._select_best_component(floor_mask, h, w)

            # ── FALLBACK: if combined gave nothing, use raw YOLO ──
            if np.sum(traversable > 0) < 500:
                yolo_clean = cv2.bitwise_and(
                    yolo_resized, (height_weight > 0.3).astype(np.uint8) * 255
                )
                if yolo_obstacle is not None:
                    obs = cv2.resize(yolo_obstacle, (w, h), interpolation=cv2.INTER_NEAREST)
                    obs = cv2.dilate(obs, np.ones((13, 13), np.uint8), iterations=1)
                    yolo_clean = cv2.bitwise_and(yolo_clean, cv2.bitwise_not(obs))
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                yolo_clean = cv2.morphologyEx(yolo_clean, cv2.MORPH_OPEN, k, iterations=1)
                yolo_clean = cv2.morphologyEx(yolo_clean, cv2.MORPH_CLOSE, k, iterations=2)
                traversable = self._select_best_component(yolo_clean, h, w)

        else:
            # ── No YOLO: classical only ──
            score_bottom = classical_weighted[int(h * 0.5):, :]
            valid = score_bottom[score_bottom > 0.01]
            if len(valid) > 100:
                std = float(np.std(valid))
                if std < 0.08:
                    pct = 80
                elif std < 0.15:
                    pct = 73
                else:
                    pct = 65
                threshold = float(np.percentile(valid, pct))
            else:
                threshold = 0.3

            floor_mask = (classical_weighted > threshold).astype(np.uint8) * 255
            floor_mask = cv2.bitwise_and(floor_mask, cv2.bitwise_not(edge_carve))

            if yolo_obstacle is not None:
                obs = cv2.resize(yolo_obstacle, (w, h), interpolation=cv2.INTER_NEAREST)
                obs = cv2.dilate(obs, np.ones((13, 13), np.uint8), iterations=1)
                floor_mask = cv2.bitwise_and(floor_mask, cv2.bitwise_not(obs))

            floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

            traversable = self._select_best_component(floor_mask, h, w)

       
        near_field_cutoff = int(h * 0.80)
        traversable[:near_field_cutoff, :] = 0

        # ── Temporal mask smoothing ──
        if self.prev_traversable is not None:
            prev_resized = cv2.resize(self.prev_traversable, (w, h), interpolation=cv2.INTER_LINEAR)
            traversable = cv2.addWeighted(traversable, 0.7, prev_resized, 0.3, 0)
            _, traversable = cv2.threshold(traversable, 127, 255, cv2.THRESH_BINARY)
        self.prev_traversable = traversable.copy()

        if return_original_res and scale < 1.0:
            traversable = cv2.resize(traversable, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            classical_score = cv2.resize(classical_score, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return traversable, classical_score
