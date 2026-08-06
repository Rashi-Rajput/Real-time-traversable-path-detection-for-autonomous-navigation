#!/usr/bin/env python3

import cv2
from ultralytics import YOLO


MODEL_PATH = "yolo26n-sem-ade20k.pt"
VIDEO_PATH = "vid4.mp4"        
OUTPUT_PATH = "videos/semantic_output.mp4"

IMG_SIZE = 1024             
DEVICE = "cpu"                 
CONF = 0.25

print("Loading model...")
model = YOLO(MODEL_PATH)

    
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

print("Processing video...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        task="semantic",
        imgsz=IMG_SIZE,
        device=DEVICE,
        conf=CONF,
        verbose=False
    )

    result = results[0]

    # Visualization
    annotated = result.plot()

    writer.write(annotated)

    cv2.imshow("YOLO26 Semantic Segmentation", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Finished.")
print("Saved:", OUTPUT_PATH)    