import cv2
import numpy as np
from ultralytics import YOLO


class YOLOSegmenter:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.traversable_classes = {
            3, 6, 9, 11, 13, 29, 46, 52, 91, 94
        }
        self.obstacle_classes = {
            0, 1, 4, 5, 7, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50,
            51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
            66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93, 95, 96,
            97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
            133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
            145, 146, 147, 148, 149
        }

    def segment(self, frame, imgsz=320, conf=0.35):
        h, w = frame.shape[:2]
        results = self.model(frame, imgsz=imgsz, conf=conf, verbose=False, device="cpu")
        traversable_mask = np.zeros((h, w), dtype=np.uint8)
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)

        if len(results) > 0 and results[0] is not None:
            r = results[0]

            if r.masks is not None and hasattr(r.masks, "data") and r.masks.data is not None:
                masks = r.masks.data.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(len(masks), dtype=int)
                for mask, cls in zip(masks, classes):
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    if cls in self.traversable_classes:
                        traversable_mask = cv2.bitwise_or(traversable_mask, mask_binary)
                    elif cls in self.obstacle_classes:
                        obstacle_mask = cv2.bitwise_or(obstacle_mask, mask_binary)
            elif hasattr(r, "semantic_mask") and r.semantic_mask is not None:
                sem_mask = r.semantic_mask.data.cpu().numpy()
                for cls_id in self.traversable_classes:
                    traversable_mask[sem_mask == cls_id] = 255
                for cls_id in self.obstacle_classes:
                    obstacle_mask[sem_mask == cls_id] = 255

        return traversable_mask, obstacle_mask
