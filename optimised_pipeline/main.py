import cv2
import numpy as np
import time
import os
from yolo_segmenter import YOLOSegmenter
from classical_pipeline import ClassicalPipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "videos","vid0.mp4")


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        return

    model_path = os.path.join(SCRIPT_DIR, "yolo26n-sem-ade20k.pt")
    segmenter = YOLOSegmenter(model_path)
    classical = ClassicalPipeline(process_width=640)

    yolo_interval = 3
    yolo_width = 320
    frame_count = 0
    prev_time = time.time()
    fps = 0.0

    cached_yolo_trav = None
    cached_yolo_obs = None

    while True:
        ret, image = cap.read()
        if not ret:
            break

        frame_count += 1

        proc_w = classical.process_width
        proc_h = int(image.shape[0] * proc_w / image.shape[1])
        proc_image = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        # Run YOLO every N frames, cache results
        if frame_count % yolo_interval == 0:
            yolo_h = int(image.shape[0] * yolo_width / image.shape[1])
            yolo_image = cv2.resize(image, (yolo_width, yolo_h), interpolation=cv2.INTER_AREA)
            yolo_trav, yolo_obs = segmenter.segment(yolo_image)
            if yolo_trav is not None:
                cached_yolo_trav = cv2.resize(yolo_trav, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
            if yolo_obs is not None:
                cached_yolo_obs = cv2.resize(yolo_obs, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)

        # Always pass cached YOLO so fusion happens every frame
        traversable, _ = classical.process(
            image, cached_yolo_trav, cached_yolo_obs, return_original_res=False
        )

        result = proc_image.copy()

        # Draw contour directly (no smoothing)
        contours, _ = cv2.findContours(traversable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                cv2.drawContours(result, [largest], -1, (0, 255, 0), 2)

        # FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
        prev_time = curr_time

        cv2.putText(
            result, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )

        # Side-by-side: result + mask
        mask_vis = cv2.cvtColor(traversable, cv2.COLOR_GRAY2BGR)
        display = np.hstack([
            cv2.resize(result, (640, 480)),
            cv2.resize(mask_vis, (640, 480)),
        ])

        cv2.imshow("Traversable path", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
