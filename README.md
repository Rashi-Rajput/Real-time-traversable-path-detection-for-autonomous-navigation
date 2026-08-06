# Real-Time Traversable Path Detection

This repository contains my work on real-time traversable path detection for autonomous navigation .

The main goal of this project is to reliably detect drivable/walkable paths directly ahead of a vehicle or robot using a standard camera. It uses a **hybrid approach** that combines a lightweight YOLO semantic segmentation model with classical computer vision techniques (LAB color distance, Canny edge barriers, texture filtering, and HSV saturation). 

Running deep learning segmentation on every frame can be heavy on a CPU, so this system runs YOLO every few frames to anchor the prediction, while fast classical computer vision algorithms run on every frame to keep the path outline smooth, accurate, and responsive.

---

## 📂 Repository Structure & Folders

Here is a breakdown of the directories in this project and what each one is for:

* **`optimised_pipeline/`**  
  **The main, production-ready pipeline.** This folder contains the real-time code that fuses YOLO semantic segmentation with classical image processing. It is optimized to run smoothly on a CPU by running YOLO every 3 frames while computing lightweight color, edge, and texture maps on every frame.

* **`gaussian_mixture/`**  
  Contains experimental scripts testing **Gaussian Mixture Models (GMM)** for road color clustering. It was used during early research to explore probabilistic color modeling for ground surfaces before switching to running LAB Gaussian statistics.

* **`img_processing_only/`**  
  Contains standalone **classical computer vision experiments**. These scripts rely purely on edge detection, thresholding, and color spaces without using any deep learning models.

* **`instancesegm+img/`**  
  Experimental scripts combining **instance segmentation** masks (detecting specific objects like cars or road boundaries) with traditional image filtering techniques.

* **`segformer/`**  
  Experiments using **SegFormer** (a transformer-based semantic segmentation model). Used to compare segmentation quality and CPU performance against lightweight YOLO models.

* **`semsegm_coloring/`**  
  Utility and visualization scripts for color-coding semantic segmentation classes and mapping class IDs to visual masks.

---

## ⚡ How the Main Pipeline Works

The core detection logic lives inside `optimised_pipeline/` and works as follows:

1. **YOLO Semantic Segmentation (`yolo_segmenter.py`)**: Runs every 3 frames at a lower resolution (`320px`) on CPU to identify traversable ground classes (road, path, sidewalk) and obstacle classes (vehicles, trees, rocks, buildings).
2. **Classical Feature Extraction (`classical_pipeline.py`)**: Runs on every frame at `640px` resolution:
   * **Adaptive LAB Color Model**: Samples the immediate ground color from the bottom of the frame and builds a running Gaussian model to track surface color over time.
   * **Canny Edge Barriers**: Detects visual boundaries to prevent the path mask from bleeding into roadside grass or walls.
   * **Texture & Saturation Filtering**: Penalizes high-texture regions (rough bushes) and high-saturation regions (vibrant green foliage).
3. **Hybrid Blending**: Combines YOLO's soft prediction map with classical feature scores ($0.55 \times \text{YOLO} + 0.45 \times \text{Classical}$) and extracts the primary path contour directly ahead of the vehicle (~3–4 meters range).

---
