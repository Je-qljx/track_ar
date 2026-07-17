import sys
sys.path.insert(0, '.')
from tests.synthetic_scene import SyntheticScene
from calibration.lane_tracker import LaneFeatureTracker
import cv2
import numpy as np

scene = SyntheticScene(
    track_type='100m', camera_motion='pan',
    pan_amplitude=0.8, pan_frames=830,
    seed=42)

tracker = LaneFeatureTracker(max_width=640, max_features=400)

calib_gray = scene.render_background(0)
if calib_gray.ndim == 3:
    calib_gray = cv2.cvtColor(calib_gray, cv2.COLOR_BGR2GRAY)
tracker.set_reference(calib_gray)

print(f'First pts: {len(tracker._first_pts) if tracker._first_pts is not None else 0}')

frame = 0
for target_f in [60, 120, 180, 240, 360, 480, 600, 720, 800]:
    steps = target_f - frame
    for _ in range(steps):
        frame += 1
        img = scene.render_background(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tracker.update(gray)
    m = tracker._match_info
    print(f'Frame {frame}: first_inliers={m.get("first_inliers", 0)} drift_orig={m.get("drift_orig", False)} pts={m.get("total_pts", 0)} klt={m.get("klt_tracked", 0)} ref_cycle={m.get("ref_cycle", False)} ref_inliers={m.get("ref_inliers", 0)}')
