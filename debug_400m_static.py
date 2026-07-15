import numpy as np
import time
from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from pipeline.main_pipeline import TrackARPipeline
from tests.synthetic_scene import SyntheticScene
from tests.self_test import SPEED, _add_tracking_grid, perturb_static, R0_400M, T0_400M
import cv2

cam_K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)
r0 = R0_400M.copy()
t0 = T0_400M.copy()
spec = None  # Standard calibration

geom = TrackGeometry(track_type='400m')
pipeline = TrackARPipeline(camera_matrix=cam_K, geometry=geom)
calib_pts = geom.calibration_world_points()
w_arr = np.array([w.as_array for w in calib_pts], dtype=np.float64)
pj, _ = cv2.projectPoints(w_arr, r0, t0, cam_K, np.zeros((4, 1)))
image_pts = [ImageCoord(float(p[0, 0]), float(p[0, 1])) for p in pj]
pipeline.calibrate_from_points(calib_pts, image_pts)
track_pts = _add_tracking_grid(geom, calib_pts, spec)
pipeline.projector.set_calibration_world_pts(track_pts)

render_proj = Projector(cam_K, np.zeros((4, 1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[SPEED] * 8)

# Init
athletes = scene.update(0.0)
canvas = scene.render_background(athletes)
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
pipeline.frame_tracker.set_reference(gray)

race_len = geom.finish_distance(1)
fps = 60.0
max_frames = int(race_len / SPEED * fps) + 200

t_start = time.time()
drop_count = 0
race_started_frame = -1

for fi in range(max_frames):
    if time.time() - t_start > 120:
        print(f'TIMEOUT at fi={fi}')
        break
    rvec, tvec = perturb_static(fi, fps, r0, t0)
    render_proj.set_extrinsics(rvec, tvec)
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    detections = scene.get_detections(athletes)
    output = pipeline.process_frame(canvas, timestamp=t, external_detections=detections)
    active = [a for a in pipeline.assigner.athletes.values() if a.tracking_confidence > 0]
    if fi >= 10 and len(active) < 8:
        drop_count += 1
    if pipeline.timer.race_started and race_started_frame < 0:
        race_started_frame = fi
    if fi % 200 == 0 or fi < 50:
        n_ath = len(pipeline.assigner.athletes)
        n_fin = len(pipeline.standings.finish_times)
        info = pipeline.frame_tracker.last_match_info
        print(f'fi={fi}: athletes={n_ath}, active={len(active)}, finished={n_fin}, '
              f'dets={len(detections)}, race_started={pipeline.timer.race_started}')
    if pipeline.timer.race_finished:
        print(f'Race finished at fi={fi}')
        break

elapsed = time.time() - t_start
print(f'\nTime: {elapsed:.1f}s ({max_frames/elapsed:.0f} fps equivalent)')
print(f'Race started at frame {race_started_frame}')
print(f'Finish times: {len(pipeline.standings.finish_times)}')
expected = int(race_len / SPEED * fps)
print(f'Expected frame: {expected}')
print(f'Drop count: {drop_count}')
for lane, ft in sorted(pipeline.standings.finish_times.items()):
    print(f'  L{lane}: {ft:.3f}s')
