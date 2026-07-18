<div align="center">

# TrackAR

_Real-time AR overlay system for track & field video_

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5c3c8c?style=flat-square&logo=opencv)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00c853?style=flat-square)](https://ultralytics.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-32/32-passing-green?style=flat-square)](#running-tests)

[Overview](#overview) • [Features](#features) • [Architecture](#architecture) • [Getting Started](#getting-started) • [Usage](#usage) • [Running Tests](#running-tests) • [Technical Details](#technical-details)

</div>

## Overview

TrackAR is a real-time computer vision system that augments track & field video with on-screen graphics — lane labels, distance markers, speed readouts, and a live leaderboard — all anchored to the physical track despite camera motion.

It supports **100m straight sprints** and **IAAF-standard 400m oval** races (staggered starts, per-lane finish distances), handles **pan/tilt/zoom/dolly/boom** camera movement, and works with both standard wide shots and telephoto close-ups using calibration target mode.

The core insight: calibrate once, then the system tracks camera motion via KLT optical flow on the track surface, keeping the AR overlay perfectly aligned frame-to-frame without drift.

## Features

- **Dual track geometry** — 100m straight and IAAF 400m oval (radius 36.5m, lane width 1.22m, straight 84.39m) with per-lane stagger offsets and finish distances
- **PnP calibration** — Standard 4-point (track line intersections) or target mode (any known-size object placed anywhere on the track); correct tangent-space rectangle on 400m curves
- **Drift-free camera tracking** — KLT optical flow (640×360, 400 features) + USAC_MAGSAC homography feeds iterative PnP re-solve; no drift even across long pan sequences
- **YOLOv8 detection** — ~115 fps pipeline on RTX 5070; configurable confidence and input size; graceful fallback to dummy detector
- **Lane assignment** — Vectorized NumPy nearest-neighbor, 2-frame pending confirmation, Kalman prediction-guided matching, NMS (IoU ≥ 0.65), track-region filtering (100m spectators), fallback re-acquisition
- **Kalman filtering** — 3-state (pos/vel/acc) constant-acceleration model with adaptive noise; velocity clamp ±15 m/s; position forced to measurement
- **Occlusion-safe AR labels** — Graphics placed 2.0m ahead (default), with behind (1.0m) and lateral (0.4m) fallbacks; bbox collision ensures zero athlete overlap
- **Live leaderboard** — Per-lane finish times recorded via video timestamp; race starts when ≥2 athletes pass 0.5m; standings panel with rank/time overlay
- **Cross-frame calibration** — ORB feature matching rectifies clicks across frames when start and finish aren't visible together (telephoto setups)
- **Synthetic demo mode** — Generates full synthetic races with 8 athletes for testing without a camera

## Architecture

### Pipeline

```
Frame in → Preprocessor → Camera Tracker (KLT) → PnP Pose Update
                               ↓
                         YOLO Detection → Lane Assignment → Position Estimation
                                                                ↓
         Race Timer ⬄ Ranking ← Position Smoothing ← Edge Detection
                                                                ↓
       Occlusion Guard → Decal Render → Standings Panel → Debug Overlay
                                                               ↓
                                                          Frame out
```

Camera tracking and athlete tracking are **independent pipelines**. The KLT tracker tracks **track surface features only** — athletes are rejected as USAC_MAGSAC outliers. The resulting homography re-projects calibration reference points through the current extrinsics, then re-solves PnP to produce a drift-free 6-DOF pose update.

> [!NOTE]
> Calibration is a one-time setup. After that, the camera can freely pan, tilt, zoom, dolly, or boom. Everything stays aligned because KLT tracks the lane lines and track surface, not the athletes.

### Module Layout

| Module | Responsibility |
|--------|---------------|
| `calibration/` | Track geometry (100m/400m), PnP calibration, 3D↔2D projection, KLT & ORB tracking |
| `detection/` | YOLOv8 person detection; `DummyDetector` for testing |
| `tracking/` | Lane assignment, Kalman filtering, position estimation |
| `pipeline/` | `TrackARPipeline` orchestrator, race timer, ranking, dynamic camera, preprocessing, smoothing, edge-case detection |
| `rendering/` | AR decal rendering, leaderboard panel, occlusion-safe placement, debug overlay |
| `ui/` | OpenCV trackbar control panel (demo) |
| `media_io/` | Threaded video capture & output |
| `tests/` | Synthetic scene generator + 32-test suite |

## Getting Started

### Requirements

- Python 3.12+
- NVIDIA GPU with CUDA 12.8 (RTX 5070 verified — 12 GB VRAM, sm_120)
- See `requirements.txt` for full dependency list

### Installation

```bash
git clone <repo-url> track_ar
cd track_ar
pip install -r requirements.txt
```

YOLOv8 weights are included (`yolov8s.pt` default, plus `yolov8n.pt` and `yolov8m.pt`), or download from Ultralytics.

---

## Usage

### GUI Application (Recommended)

```bash
python trackar_gui.py
```

Provides a full Chinese-language interface with:

- Video file browser
- Track type selection (100m / 400m)
- Focal length slider (24–800mm full-frame equivalent)
- Standard 4-point calibration or **calibration target mode**
- Click-based calibration with cross-frame ORB rectification
- YOLO confidence / input-size controls
- Processing progress bar with output viewer

> [!TIP]
> **Calibration target mode** is designed for telephoto shots where start and finish aren't visible together. Place a known-size object (e.g. A4 paper) at a known position on the track, click its 4 corners, then remove it. On 400m curves, the target rectangle is computed in the local tangent plane for correct PnP.

### Synthetic Demo

```bash
# 100m straight sprint
python demo.py --track 100m

# 400m oval
python demo.py --track 400m
```

Keyboard controls:

| Key     | Action                          |
| ------- | ------------------------------- |
| `Space` | Pause / resume                  |
| `B`     | Toggle bounding boxes           |
| `O`     | Toggle AR overlay               |
| `F`     | Toggle follow-mode camera       |
| `R`     | Reset race                      |
| `Q`     | Quit                            |

### Real Video Processing

```bash
python run_real_video.py race_video.mp4 --track-type 100m
python run_real_video.py race_video.mp4 --track-type 400m --focal-mm 200
```

Options:

| Flag | Description |
|------|-------------|
| `--fx` | Focal length in pixels |
| `--focal-mm` | 35mm equivalent focal length (used to compute fx) |
| `--track-type` | `100m` or `400m` |
| `--no-yolo` | Use dummy detector instead of YOLO |
| `--model` | YOLO model path (default: `models/yolov8s.pt`) |
| `--detect-conf` | Detection confidence threshold (default: 0.25) |
| `--yolo-imgsz` | YOLO input size (default: 1280) |
| `--output` | Output video path |
| `--max-frames` | Limit number of frames to process |

## Running Tests

```bash
python tests/self_test.py
```

Test suite: **32/32 passing**

| Category | Count | Description |
|----------|-------|-------------|
| Quick calibration | 13 | Standard/target calibration at various positions and sizes for both 100m/400m |
| Full-race static | 4 | Complete races with static camera (standard + target, 100m + 400m) |
| Full-race pan | 4 | Panning camera (standard + target, 100m + 400m, within 0.2s) |
| Full-race zoom/panzoom | 3 | Zoom and combined pan-zoom (within 0.2s) |
| Full-race jitter/false-positives/dropout/noise | 4 | Stress conditions: random shake, 30 noisy detections/frame, 50% dropout, 3px calibration noise (within 0.5–1.0s) |
| Dummy detector | 1 | YOLO fallback path |
| Stress tests | 3 | Occlusion, dropout, sudden appearance |

Covers: static, pan, zoom, panzoom, moderate pan for 100m; static, pan, panzoom for 400m.

## Project Structure

```
track_ar/
├── demo.py                      # Interactive synthetic demo
├── run_real_video.py            # Real video processing entry point
├── trackar_gui.py               # Tkinter GUI (Chinese interface, full controls)
├── requirements.txt
│
├── calibration/                 # Camera calibration & track geometry
│   ├── coords.py                # Coordinate dataclasses + TrackGeometry (100m/400m)
│   ├── track_model.py           # IAAF-standard 400m oval model
│   ├── calibrator.py            # solvePnP / solvePnPRansac calibration
│   ├── projector.py             # 3D↔2D projection + real-time homography pose update
│   ├── frame_tracker.py         # ORB feature matching (cross-frame calibration)
│   └── lane_tracker.py          # KLT optical flow (per-frame pipeline)
│
├── detection/
│   └── detector.py              # YOLODetector, DummyDetector, Detection dataclass
│
├── tracking/
│   ├── lane_assigner.py         # Lane-to-athlete assignment + trajectory management
│   ├── kalman.py                # 3-state (pos/vel/acc) Kalman filter
│   └── position_estimator.py    # d_m and speed estimation
│
├── pipeline/
│   ├── main_pipeline.py         # TrackARPipeline orchestrator
│   ├── timing.py                # Video-timestamp race timer
│   ├── ranking.py               # Per-frame rank computation
│   ├── dynamic_camera.py        # Follow-mode camera look-at
│   ├── preprocessor.py          # Frame preprocessing (square resize)
│   ├── smoother.py              # EMA position smoothing
│   └── edge_cases.py            # Fall detection, speed anomalies
│
├── rendering/
│   ├── standings.py             # On-screen leaderboard
│   ├── decal_renderer.py        # AR overlay with alpha blending
│   ├── graphic_factory.py       # Rank/time texture generation
│   ├── occlusion_guard.py       # Safe anchor placement (ahead/behind/lateral)
│   └── debug_overlay.py         # Bbox and anchor visualization
│
├── ui/
│   └── control_panel.py         # OpenCV trackbar panel (demo)
│
├── media_io/
│   └── video_io.py              # Threaded capture & output
│
├── utils/
│   └── logger.py                # CSV metrics logging
│
├── tests/
│   ├── self_test.py             # 32-test comprehensive suite
│   ├── synthetic_scene.py       # Synthetic track with 8 athletes + Perlin noise
│   ├── synth_video.py           # Synthetic video generator
│   └── stress_test.py           # Edge-case stress tests
│
├── scripts/                     # Integration verification scripts
│
└── models/                      # YOLOv8 model weights
    ├── yolov8s.pt               # (default, ~115 fps)
    ├── yolov8n.pt
    └── yolov8m.pt
```

---

## Technical Details

| Component | Detail |
|-----------|--------|
| **GPU** | RTX 5070 Laptop (12 GB, sm_120), PyTorch 2.12 nightly, CUDA 12.8 |
| **YOLO pipeline** | ~115 fps at yolov8s, configurable imgsz (default 1280), conf (default 0.25), iou (0.5); graceful fallback to DummyDetector |
| **Camera tracking** | KLT at 640×360, 400 features (goodFeaturesToTrack), quality=0.005, min_distance=3, redetect every 60 frames; USAC_MAGSAC (reproj 3.0) homography + PnP re-solve |
| **PnP** | solvePnPRansac (ITERATIVE), ~330-point tracking grid (400m), ~176 points (100m); no extrinsic guess to avoid local minima |
| **400m track model** | IAAF standard: inner-edge radius 36.5m, lane width 1.22m, straight 84.39m; per-lane curve arcs, stagger offsets, finish distances |
| **Calibration target** | Tangent-space rectangle for correct PnP on curved 400m tracks; supports any lane/dm position |
| **Race timer** | Video-timestamp (not wall clock); starts at ≥2 athletes past 0.5m; stops when all 8 lanes finished |
| **Lane assignment** | Vectorized NumPy nearest neighbor; 2-frame pending confirmation; IoU NMS ≥ 0.65; track-region filter (100m spectator rejection); fallback re-acquisition |
| **Kalman filter** | 3-state constant-acceleration; velocity clamp ±15 m/s; position forced to measurement; adaptive noise by tracking confidence |
| **Occlusion guard** | Anchors 2.0m ahead (default), 1.0m behind, 0.4m lateral; bbox collision check ensures zero athlete overlap |

---

## Technical Design

For detailed design documentation covering camera placement, occlusion rules, 3D rendering pipeline, and edge cases, see [`track-ar-system-design.md`](track-ar-system-design.md).
