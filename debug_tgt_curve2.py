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
spec = (60.0, 1, 0.420, 0.297)

geom = TrackGeometry(track_type='400m')
pipeline = TrackARPipeline(camera_matrix=cam_K, geometry=geom)

dm_t, lane_t, w_t, h_t = spec
cy = geom.lane_center_y(lane_t)
calib_pts = [
    WorldCoord(dm_t - w_t/2, cy - h_t/2, 0.0),
    WorldCoord(dm_t + w_t/2, cy - h_t/2, 0.0),
    WorldCoord(dm_t + w_t/2, cy + h_t/2, 0.0),
    WorldCoord(dm_t - w_t/2, cy + h_t/2, 0.0),
]
w_arr = np.array([w.as_array for w in calib_pts], dtype=np.float64)
pj, _ = cv2.projectPoints(w_arr, r0, t0, cam_K, np.zeros((4, 1)))
image_pts = [ImageCoord(float(p[0, 0]), float(p[0, 1])) for p in pj]
pipeline.calibrate_from_points(calib_pts, image_pts)
print(f'Calib err: {pipeline.calibrator.get_projection_error(calib_pts, image_pts):.3f}px')

track_pts = _add_tracking_grid(geom, calib_pts, spec)
pipeline.projector.set_calibration_world_pts(track_pts)
print(f'Grid pts: {len(track_pts)}')

render_proj = Projector(cam_K, np.zeros((4, 1)))
render_proj.set_extrinsics(r0.copy(), t0.copy())
scene = SyntheticScene(render_proj, geom, speeds=[SPEED] * 8)

# Render first frame and check detections
athletes = scene.update(0.0)
canvas = scene.render_background(athletes)
detections = scene.get_detections(athletes)
print(f'Detections: {len(detections)}')
for d in detections:
    print(f'  bbox={d.bbox}, conf={d.confidence:.2f}, bottom_center={d.bottom_center}')

# Check if detections are in track region
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
pipeline.frame_tracker.set_reference(gray)
print(f'Init features: {len(pipeline.frame_tracker._pts) if pipeline.frame_tracker._pts is not None else 0}')

# Process first frame
output = pipeline.process_frame(canvas, timestamp=0.0, external_detections=detections)
print(f'After frame 0: athletes={len(pipeline.assigner.athletes)}')
for a in pipeline.assigner.athletes.values():
    print(f'  L{a.lane}: d_m={a.d_m:.1f}, conf={a.tracking_confidence:.2f}')
print(f'  render_proj rvec={render_proj.rvec.ravel()}')
print(f'  pipe.proj  rvec={pipeline.projector.rvec.ravel() if pipeline.projector.rvec is not None else None}')

# Process frames 1-100
t_start = time.time()
for fi in range(1, 1201):
    rvec, tvec = perturb_static(fi, 60.0, r0, t0)
    render_proj.set_extrinsics(rvec, tvec)
    t = fi / 60.0
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    detections = scene.get_detections(athletes)
    output = pipeline.process_frame(canvas, timestamp=t, external_detections=detections)
    if fi % 30 == 0:
        active = [a for a in pipeline.assigner.athletes.values() if a.tracking_confidence > 0]
        pending = pipeline.assigner._pending
        info = pipeline.frame_tracker.last_match_info
        n_pts = info.get('total_pts', 0)
        method = info.get('method', '?')
        cfail = pipeline.frame_tracker._consecutive_failures
        print(f'  fi={fi}: athletes={len(pipeline.assigner.athletes)}, active={len(active)}, '
              f'pending={len(pending)}, pts={n_pts}, meth={method}, fail={cfail}')
        print(f'    pipe rvec={pipeline.projector.rvec.ravel()}')
    if pipeline.timer.race_finished:
        print(f'Race finished at fi={fi}')
        break

print(f'Time: {time.time()-t_start:.1f}s')
print(f'Athletes: {len(pipeline.assigner.athletes)}')
for a in pipeline.assigner.athletes.values():
    print(f'  L{a.lane}: d_m={a.d_m:.1f}, conf={a.tracking_confidence:.2f}, coast={a.coast_count}')
