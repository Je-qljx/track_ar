# TrackAR

*Real-time AR overlay system for track & field video broadcasting*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5c3c8c?style=flat-square&logo=opencv)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00c853?style=flat-square)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-35/35-passing-green?style=flat-square)](#running-tests)

TrackAR is a computer vision system that adds real-time AR overlays to track & field video feeds. It supports **100m straight sprints** and **IAAF-standard 400m oval** races, with calibration target mode for telephoto lenses, real-time camera tracking (pan/tilt/zoom/dolly/boom), occlusion-safe graphics placement, and on-screen leaderboard standings.

---

## Features

- **Dual track support** — 100m straight sprint and IAAF-standard 400m oval with staggered starts and per-lane finish distances
- **PnP camera calibration** — Standard 4-point calibration (start/finish line intersections) or target mode (any known-size object at any position on the track)
- **Real-time camera tracking** — KLT optical flow tracks track-surface features frame-to-frame; USAC_MAGSAC homography feeds PnP to update 6-DOF camera pose without drift
- **Athlete detection** — YOLOv8 person detection with fallback dummy detector for synthetic testing
- **Lane assignment** — Vectorized nearest-neighbor matching with 2-frame pending confirmation, Kalman prediction-guided search, non-maximum suppression (NMS), and track-region filtering
- **Kalman filtering** — 3-state (pos/vel/acc) constant-acceleration model with adaptive measurement noise
- **Occlusion-safe graphics** — Anchors placed ahead, behind, or laterally to ensure AR labels never cover athletes
- **Leaderboard standings** — Per-lane finish-time tracking with video-timestamp-based race timer
- **Cross-frame calibration** — Calibrate across multiple frames when start/finish aren't visible together; ORB feature matching rectifies points before PnP
- **Synthetic demo** — Full synthetic scene with Perlin-noise background for testing without a camera

---

## Architecture

The processing pipeline runs per frame:

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

Camera tracking and athlete tracking are independent. The KLT feature tracker operates on background features only (athletes are rejected as USAC_MAGSAC outliers). The homography feeds `Projector.track_homography()`, which re-projects calibration reference points through the current extrinsics, warps via the tracked homography, then re-solves PnP — producing **drift-free 6-DOF pose updates** aligned to the original calibration.

> **Note:** Calibration happens once. After that, the camera can freely pan, tilt, zoom, dolly, or boom. The system tracks camera motion entirely via KLT on lane lines and track surface features, keeping the overlay aligned.

---

## Getting Started

### Requirements

- Python 3.12
- NVIDIA GPU with CUDA 12.8 recommended for YOLO acceleration (RTX 5070 verified)
- See `requirements.txt` for full dependency list

### Installation

```bash
git clone <repo-url> track_ar
cd track_ar
pip install -r requirements.txt
```

YOLOv8 weights are included (`yolov8s.pt`, `yolov8n.pt`, `yolov8m.pt`), or download from Ultralytics.

---

## Usage

### GUI Application (Recommended)

```bash
python trackar_gui.py
```

The GUI provides:
- Video file browser with track type selection (100m / 400m)
- Camera focal length slider
- Standard 4-point calibration or calibration target mode (set width, height, distance mark, lane)
- Click-based calibration with cross-frame ORB rectification
- YOLO toggle
- Processing progress and output viewer

**Calibration target mode** is recommended for telephoto shots where start and finish aren't visible together. Place a known-size object (A4 paper, cardboard box) at a known position before the race, click its 4 corners, then remove it — calibration complete.

**Standard mode** works when a single frame shows both start and finish lines. Click the 4 track-line intersections (Start×Lane1, Start×Lane8, Finish×Lane1, Finish×Lane8).

### Synthetic Demo

```bash
# 100m straight sprint
python demo.py --track 100m

# 400m oval
python demo.py --track 400m
```

Keyboard controls during demo:

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `B` | Toggle detection bounding boxes |
| `O` | Toggle AR overlay |
| `F` | Toggle follow-mode camera |
| `R` | Reset race |
| `Q` | Quit |

### Real Video Processing

```bash
python run_real_video.py --video path/to/video.mp4 --track 100m
python run_real_video.py --video path/to/video.mp4 --track 400m --fx 2000
```

Options include `--fx` (focal length), `--no-yolo` (use dummy detector), `--output` (output path), `--max-frames` (limit frames).

---

## Running Tests

```bash
python tests/self_test.py
```

Test suite: **35/35 passing**

| Category | Count | Description |
|----------|-------|-------------|
| Quick calibration checks | 13 | Standard/target calibration at various positions, sizes, and camera angles for both 100m and 400m |
| Full-race static | 10 | Complete races with static camera (standard + target + side view) |
| Full-race pan | 4 | Panning camera during race (standard + target, 100m + 400m) |
| Full-race 400m zoom | 2 | Zoom during 400m race |
| Full-race boom | 2 | Boom up/down during 100m race |
| Full-race zoom/dolly | 4 | Zoom and dolly during 100m race |

> **Note:** 100m zoom and dolly tests have relaxed tolerances (1.5s / 0.5s) due to PnP depth ambiguity on near-field cameras (z=36m). 400m tests at z=90m pass within 0.2s for all motion types.

---

## Project Structure

```
track_ar/
├── demo.py                      # Interactive synthetic demo
├── run_real_video.py            # Real video processing entry point
├── trackar_gui.py               # Tkinter GUI application
├── requirements.txt
│
├── calibration/                 # Camera calibration & track geometry
│   ├── coords.py                # Coordinate dataclasses + TrackGeometry
│   ├── track_model.py           # IAAF 400m oval track model
│   ├── calibrator.py            # PnP calibration solver
│   ├── projector.py             # 3D↔2D projection + real-time homography tracking
│   ├── frame_tracker.py         # ORB feature matching (cross-frame rectification)
│   └── lane_tracker.py          # KLT optical flow tracker (pipeline)
│
├── detection/
│   └── detector.py              # YOLODetector / DummyDetector + Detection dataclass
│
├── tracking/
│   ├── lane_assigner.py         # Lane-to-athlete assignment + trajectory management
│   ├── kalman.py                # 3-state position/velocity/acceleration Kalman filter
│   └── position_estimator.py    # Distance-along-track and speed estimation
│
├── pipeline/
│   ├── main_pipeline.py         # TrackARPipeline orchestrator
│   ├── timing.py                # Video-timestamp-based race timer
│   ├── ranking.py               # Per-frame rank computation
│   ├── dynamic_camera.py        # Follow-mode camera look-at control
│   ├── preprocessor.py          # Frame preprocessing
│   ├── smoother.py              # EMA-based position smoothing
│   └── edge_cases.py            # Fallen-athlete and anomaly detection
│
├── rendering/
│   ├── standings.py             # On-screen leaderboard panel
│   ├── decal_renderer.py        # AR overlay rendering with alpha blending
│   ├── graphic_factory.py       # Rank/time texture generation
│   ├── occlusion_guard.py       # Safe anchor placement (ahead/behind/lateral)
│   ├── depth_sorter.py          # Z-ordering by distance from camera
│   └── debug_overlay.py         # Detection bbox and anchor visualization
│
├── ui/
│   └── control_panel.py         # OpenCV trackbar-based control panel
│
├── media_io/
│   └── video_io.py              # Threaded video capture and output
│
├── utils/
│   └── logger.py                # CSV metrics logging
│
├── tests/
│   ├── self_test.py             # 35-test comprehensive suite
│   ├── synthetic_scene.py       # Synthetic track scene with Perlin-noise background
│   ├── synth_video.py           # Synthetic video generator
│   ├── stress_test.py           # Edge-case stress tests
│   └── test_occlusion_guard.py  # Occlusion guard unit tests
│
├── scripts/                     # Integration verification scripts
│
└── models/                      # YOLOv8 model weights
    ├── yolov8n.pt
    ├── yolov8s.pt
    └── yolov8m.pt
```

---

## Key Technical Details

| Component | Detail |
|-----------|--------|
| **GPU** | RTX 5070 Laptop (12 GB, sm_120) — PyTorch 2.12 nightly + CUDA 12.8 |
| **YOLO pipeline** | ~115 fps at yolov8s; graceful fallback to `DummyDetector` |
| **Camera tracking** | KLT at 640×360, 400 features, quality=0.005, min_distance=3px, redetect every 60 frames; USAC_MAGSAC (3.0 reproj) homography |
| **PnP** | `solvePnPRansac` (ITERATIVE) with dense ~330-point tracking grid; no extrinsic guess to avoid local minima |
| **400m track model** | IAAF standard: inner-edge radius 36.5 m, lane width 1.22 m, straight 84.39 m; per-lane curve arcs, stagger offsets, and finish distances |
| **Race timer** | Video-timestamp-based (not wall clock); starts when ≥2 athletes pass 0.5 m; stops when all 8 lanes finished |
| **Lane assignment** | Vectorized NumPy nearest-neighbor; 2-frame pending-track confirmation; NMS (IoU ≥ 0.85); track-region filtering for spectator rejection (100m); fallback re-acquisition with relaxed thresholds (400m) |
| **Occlusion guard** | Places graphic anchors 2.0 m ahead (default), with behind (1.0 m) and lateral (0.4 m) fallbacks; bbox collision check ensures zero athlete overlap |

---

## Technical Design

For detailed design documentation (camera placement, occlusion rules, 3D rendering pipeline, edge cases), see [`track-ar-system-design.md`](track-ar-system-design.md).
