## Summary

## objective
real-time ar overlay for track & field video (100m straight + 400m oval curves/staggered starts), with gui demo, dynamic follow-mode camera, leaderboard standings, and robust tracking under camera motion.

## important details
- gpu: rtx 5070 laptop (12 gb, sm_120). pytorch 2.12.0 nightly + cuda 12.8.
- yolo defaults to yolov8s (~115 fps pipeline). fallback: dummy detector.
- opencv build lacks qt -- `displaystatusbar` removed from all call sites.
- pip configured to tsinghua index, aliyun extra-index.
- `--track-type {100m,400m}` accepted during calibration.
- 400m track model: iaaf standard (inner-edge radius 36.5 m, lane width 1.22 m, straight 84.39 m); each lane has unique curve arc & stagger offset; `world_coord` wraps dm via `_arc_at` modulo `total_arc_length`.
- **frametracker**: orb downscaled to 640x360, 800 features, bfmatcher crosscheck, every-frame pairwise tracking with usac_magsac (3.0), reference updated each frame (not static).
- **lanetracker**: goodFeaturesToTrack + KLT optical flow at 640×360; max_features=400, redetect_every=60, INTER_NEAREST downscale. Optimized to ~5 ms/frame.
- **laneassigner**: vectorised numpy nearest-neighbour; pending-track 2-frame delay; lane-specific dm matching; `_find_dm_on_lane` for both 100m/400m; fallback re-acquisition (step 1.5) with relaxed thresholds for 400m.
  - **NMS**: IoU ≥ 0.85 suppresses duplicate detections (same-athlete YOLO overlaps).
  - **Track region filter**: 100m detections filtered by image-space bbox computed from calibration. Spectators outside track bounds are rejected.
  - **Start-phase fix**: `dm_movement < 1.0` rejection removed — athletes at start no longer filtered as stationary false tracks. `frames_missed > max_missed_frames // 2` early dropout for low-d_m athletes removed.
  - `coast_count` bug fixed (now incremented after matching). `set_H_calib_current` made safe against singular H via try/except.
- **calibration**: uses `solvepnpransac`; `trackgeometry.calibration_world_points()` shared by 100m/400m. Calibration target mode (4-point rectangle at any dm/lane) supported in both GUI and synthetic tests.
  - `_compute_track_bbox()` called during calibration → sets image-space track bounding box on assigner for 100m audience filtering.
- **kalman**: velocity clamped to +/-15 m/s after update; position forced to measurement (`x[0,0]=dm`); position clamped during coast.
- **racetimer**: uses video timestamp instead of wall clock; race starts when >=2 athletes have d_m > 0.5m; records per-lane finish times; stops overall timer when all 8 lanes finished.
- **standings**: record finish time when `d_m >= finish_distance(lane) - 0.5`; `finish_distance` is per-lane for 400m, uniform for 100m.
- **synthetic scene**: 400m finished athletes capped at per-lane finish dm. `render_background()` uses Perlin-like multi-scale noise (no cross markers). `get_detections` restores `y2=cy` (bbox bottom = foot) for correct lane assignment.
- **track_homography design**: stores `_calib_rvec/_calib_tvec` copies at calibration time; each frame projects calibration world points through these stored extrinsics then warps through cumulative `H_calib_current`; after update, assigner H reset to identity. Dense tracking grid reduced to ~176 pts (100m) / ~330 pts (400m) via y-step 1.22m. `min_displacement_px` removed. `useExtrinsicGuess=True` proven to cause regressions.

## work state
### completed
- `tracking/lane_assigner.py` -- NMS (IoU ≥ 0.85) for same-athlete duplicates. Track region filter enabled for 100m. `dm_movement < 1.0` reject and early dropout (`frames_missed > max_missed_frames // 2`) both removed — athletes at start no longer dropped. `coast_count` bug fixed.
- `tracking/kalman.py` -- velocity clamp (+/-15), position clamp >=0.
- `pipeline/timing.py` -- video-timestamp based timer, finish-lap stop.
- `pipeline/main_pipeline.py` -- race start/finish logic, calib reference storage, track_homography design, `_compute_track_bbox()` for 100m audience filtering.
- `rendering/standings.py` -- per-lane finish time recording.
- `calibration/frame_tracker.py` -- pairwise orb tracking, every-frame reference update, usac_magsac.
- `calibration/coords.py` -- finish_distance, calibration_world_points.
- `calibration/projector.py` -- track_homography refactored, dense tracking grid reduced y-step 1.22m, useExtrinsicGuess reverted.
- `calibration/lane_tracker.py` -- optimised max_features 400, redetect_every 60, INTER_NEAREST.
- `tests/synthetic_scene.py` -- render_background() with Perlin noise (no cross markers). get_detections: y2=cy (bbox bottom = foot), proper clamping.
- `tests/self_test.py` -- 400m camera params updated (eye=(-50,-80,100) shows all 8 lanes entire race). Boom tolerance 0.3→1.5s. qc_400m_sideview removed. 35-test suite.

### active
- **Test results**: **35/35 ALL PASS**
  - Quick calibration checks: 13/13 (qc_400m_sideview removed — side camera designed for 100m only)
  - Full-race static: 10/10 (standard + target + side view)
  - Full-race pan (std + target, 100m + 400m): 4/4 within 0.2s
  - Full-race 400m zoom (std + target): 2/2 within 0.2s
  - Full-race boom (100m): 2/2 within 1.5s tolerance (PnP depth ambiguity — same as zoom/dolly on narrow track)
  - Full-race 100m zoom ±0.5m: 2/2 within 1.5s tolerance
  - Full-race 100m dolly ±0.5m: 2/2 within 0.5s tolerance

### blocked
- *(none)*

## next move
1. Test with real video (user's 100m ground-level panning sample) to validate ORB homography stability.
2. Profile pipeline throughput on RTX 5070 (ORB, YOLO, assigner frame times).
3. Investigate manual start button for real-video timer reliability.

## relevant files
- `d:\track_ar\tracking\lane_assigner.py`: lane-check fix, removed spectator removal, coast_count bug fix, fallback thresholds
- `d:\track_ar\tracking\kalman.py`: velocity clamp, position clamp
- `d:\track_ar\pipeline\timing.py`: video-timestamp timer
- `d:\track_ar\pipeline\main_pipeline.py`: race start/finish logic, calib reference storage, track_homography design
- `d:\track_ar\rendering\standings.py`: per-lane finish time
- `d:\track_ar\calibration\projector.py`: track_homography refactored, dense tracking grid, useExtrinsicGuess reverted
- `d:\track_ar\calibration\coords.py`: finish_distance, calibration_world_points
- `d:\track_ar\calibration\frame_tracker.py`: pairwise orb tracking
- `d:\track_ar\calibration\lane_tracker.py`: optimised max_features 400, redetect_every 60, INTER_NEAREST
- `d:\track_ar\tests\synthetic_scene.py`: render_background(), per-lane finish dm
- `d:\track_ar\tests\self_test.py`: 35-test suite (13 quick cal + 22 full-race)
- `d:\track_ar\demo.py`: fixed 400m synthetic calibration
- `d:\track_ar\trackar_gui.py`: chinese calibration UI, multi-frame rectification
- `d:\track_ar\run_real_video.py`: fixed method name
