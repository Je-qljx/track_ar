import sys, os, time as _time
sys.path.insert(0, '.')
import numpy as np
import cv2

from calibration.coords import TrackGeometry
from calibration.calibrator import Calibrator
from calibration.projector import Projector
from calibration.lane_tracker import LaneFeatureTracker
from tracking.lane_assigner import LaneAssigner
from tracking.position_estimator import PositionEstimator
from pipeline.timing import RaceTimer
from tests.synthetic_scene import SyntheticScene, SynthAthleteState
from detection.detector import Detection

SPEED = 9.5
pan_amplitude = 0.8
pan_frames = 830

cam_K = np.array([[2300, 0, 960], [0, 2300, 540], [0, 0, 1]], dtype=np.float64)
cam_dist = np.zeros((4, 1), dtype=np.float64)

r0 = np.array([[0.3], [0.0], [0.0]], dtype=np.float64)
t0 = np.array([[0.0], [0.0], [50.0]], dtype=np.float64)

geom = TrackGeometry(track_type='100m', camera_matrix=cam_K, image_size=(1920, 1080))

def perturb_pan_wide(fi, fps, r0, t0):
    r = r0.copy().astype(np.float64)
    frac = fi / max(pan_frames, 1)
    r[1, 0] = r0[1, 0] - pan_amplitude * frac
    return r, t0.copy()

calibrator = Calibrator(camera_matrix=cam_K, image_size=(1920, 1080))
projector = Projector(cam_K, cam_dist, image_size=(1920, 1080))
tracker = LaneFeatureTracker(max_width=640, max_features=400)
assigner = LaneAssigner(geom, projector)
timer = RaceTimer()

calib_pts = geom.calibration_world_points()
ret = calibrator.calibrate(calib_pts, r0, t0, target_info=None)
print(f"Calibrated: {ret}")

projector.set_extrinsics(calibrator.rvec.copy(), calibrator.tvec.copy())
calib_rvec = calibrator.rvec.copy()
calib_tvec = calibrator.tvec.copy()

render_proj = Projector(cam_K, np.zeros((4, 1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[SPEED] * 8)
fps = 60.0
max_frames = int(geom.finish_distance(1) / SPEED * fps) + 200

frame0 = scene.render_background([])
gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
tracker.set_reference(gray0)
print(f"Tracker reference set, pts={len(tracker._first_pts)}")

projector.set_calibration_world_pts(geom.calibration_world_points())

assigner.set_H_calib_current(np.eye(3, dtype=np.float64))
bbox = calibrator._compute_track_bbox(geom)
if bbox:
    assigner.set_track_bbox_100m(bbox)

tracker._first_klt_win_size = (63, 63)
tracker._first_klt_max_level = 5

finish_times = {}
t_start = _time.time()
drop_count = 0
race_started_frame = -1

for fi in range(max_frames):
    if _time.time() - t_start > 120:
        print("TIMEOUT")
        break
    rvec, tvec = perturb_pan_wide(fi, fps, r0, t0)
    render_proj.set_extrinsics(rvec, tvec)
    t = fi / fps
    athletes_data = scene.update(t)
    canvas = scene.render_background(athletes_data)
    detections = scene.get_detections(athletes_data)

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    tracker.update(gray)
    H = tracker.H_calib_current
    assigner.set_H_calib_current(H)
    if calib_rvec is not None:
        projector.rvec = calib_rvec
        projector.tvec = calib_tvec
    H_pnp = H
    if hasattr(tracker, '_scale') and abs(tracker._scale - 1.0) > 0.01:
        s = 1.0 / tracker._scale
        S = np.diag([s, s, 1.0]).astype(np.float64)
        H_pnp = S @ H @ np.linalg.inv(S)
    pose_updated = projector.track_homography(H_pnp)
    if pose_updated:
        assigner.set_H_calib_current(np.eye(3, dtype=np.float64))

    dets = [Detection(d.x1, d.y1, d.x2, d.y2, d.confidence, d.class_id) for d in detections]
    assigner.process_frame(dets, frame_dt=1.0/fps)

    positions = []
    for athlete_id, athlete in assigner.athletes.items():
        if athlete.tracking_confidence <= 0:
            continue
        pos = athlete.kalman.get_position()
        speed = athlete.kalman.get_speed()
        positions.append(type('pos', (), {
            'lane': athlete.lane, 'athlete_id': athlete_id,
            'd_m': float(pos[0]), 'speed_mps': float(speed),
            'confidence': athlete.tracking_confidence,
            'timestamp': t})())

    if not timer.race_started:
        past = sum(1 for p in positions if p.d_m > 0.5 and p.speed_mps > 2.0 and p.confidence > 0)
        if past >= 2:
            timer.start_race(t)
            race_started_frame = fi
            print(f"Race started at frame {fi}, t={t:.2f}s, past={past}")
            for p in positions:
                print(f"  frame={fi} lane={p.lane} d_m={p.d_m:.3f} speed={p.speed_mps:.3f} conf={p.confidence:.3f}")

    for p in positions:
        finish_dm = geom.finish_distance(p.lane)
        if p.d_m >= finish_dm - 0.5 and p.lane not in finish_times and timer.race_started:
            finish_times[p.lane] = timer.get_elapsed(t)
            if timer.race_finished:
                break

    if fi <= 30:
        info = tracker._match_info
        print(f"Frame {fi}: first_inl={info.get('first_inliers', 0)} "
              f"drift_orig={info.get('drift_orig', False)} pts={info.get('total_pts', 0)} "
              f"pose_upd={pose_updated} active={sum(1 for a in assigner.athletes.values() if a.tracking_confidence>0)}")

    if timer.race_finished:
        break

print(f"\nResults: {len(finish_times)}/8 finished, frames={fi}")
print(f"drop_count={drop_count}, race_started_frame={race_started_frame}")
for lane in sorted(finish_times.keys()):
    print(f"  Lane {lane}: {finish_times[lane]:.3f}s")
