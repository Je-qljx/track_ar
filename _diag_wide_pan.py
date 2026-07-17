import sys; sys.path.insert(0, 'D:/track_ar')
import numpy as np
import cv2

from calibration.coords import TrackGeometry, ImageCoord, WorldCoord
from calibration.projector import Projector
from pipeline.main_pipeline import TrackARPipeline
from tests.synthetic_scene import SyntheticScene

SPEED = 9.5
K = np.array([[700, 0, 960], [0, 700, 540], [0, 0, 1]], dtype=np.float64)
R0_100M = np.array([[-1.9865], [-0.7462], [-0.4312]], dtype=np.float64)
T0_100M = np.array([[-40.827], [-14.574], [36.127]], dtype=np.float64)


def perturb_pan_wide(f, fps, r0, t0):
    r = r0.copy()
    t = t0.copy()
    t_sec = f / fps
    progress = min(t_sec / (100.0 / SPEED), 1.0)
    r[1, 0] += 0.8 * progress
    return r, t


def _add_tracking_grid(geom, calib_pts, target_spec=None):
    pts = list(calib_pts)
    is_400m = geom._model is not None
    for lane in range(1, 9):
        for dm in np.arange(0.0, geom.length + 1, 10.0):
            if is_400m:
                wc = geom.world_coord(lane, dm)
            else:
                wc = WorldCoord(dm, geom.lane_center_y(lane), 0.0)
            if target_spec is not None:
                dm_t, lane_t, _, _ = target_spec
                if lane == lane_t and abs(dm - dm_t) < 1.0:
                    continue
            pts.append(wc)
    return pts


def compute_in_bounds(H, calib_world_pts, rvec, tvec, cam_K, dist, img_size):
    if H is None or calib_world_pts is None or rvec is None or tvec is None:
        return 0
    calib_2d, _ = cv2.projectPoints(calib_world_pts, rvec, tvec, cam_K, dist)
    current_2d = cv2.perspectiveTransform(calib_2d, H)
    margin = 200
    w, h = img_size
    pts_2d = current_2d.reshape(-1, 2)
    in_bounds = ((pts_2d[:, 0] >= -margin) & (pts_2d[:, 0] < w + margin) &
                 (pts_2d[:, 1] >= -margin) & (pts_2d[:, 1] < h + margin))
    return int(np.sum(in_bounds))


cv2.setRNGSeed(42)
np.random.seed(42)

print("=== 100m/pan_wide/extreme per-frame diagnostics ===")

geom = TrackGeometry(track_type="100m")
pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)

calib_pts = geom.calibration_world_points()
w_arr = np.array([w.as_array for w in calib_pts], dtype=np.float64)
proj, _ = cv2.projectPoints(w_arr, R0_100M, T0_100M, K, np.zeros((4, 1)))
image_pts = [ImageCoord(float(p[0, 0]), float(p[0, 1])) for p in proj]
pipeline.calibrate_from_points(calib_pts, image_pts)

track_pts = _add_tracking_grid(geom, calib_pts)
pipeline.projector.set_calibration_world_pts(track_pts)

render_proj = Projector(K, np.zeros((4, 1)))
render_proj.set_extrinsics(R0_100M.copy(), T0_100M.copy())
scene = SyntheticScene(render_proj, geom, speeds=[SPEED] * 8)

fps = 60.0
max_frames = int(100.0 / SPEED * fps) + 200
race_len = geom.finish_distance(1)

print(f"max_frames={max_frames}, SPEED={SPEED}, race_len={race_len}m\n")

calib_err = pipeline.calibrator.get_projection_error(calib_pts, image_pts)
print(f"Calibration error: {calib_err:.4f} px\n")

last_conf: dict[int, float] = {lane: 0.5 for lane in range(1, 9)}
dropped_log: set[int] = set()
removed_log: set[int] = set()
finish_log: set[int] = set()

for fi in range(max_frames):
    rvec, tvec = perturb_pan_wide(fi, fps, R0_100M, T0_100M)
    render_proj.set_extrinsics(rvec, tvec)
    t = fi / fps
    athletes = scene.update(t)
    canvas = scene.render_background(athletes)
    detections = scene.get_detections(athletes)
    pipeline.process_frame(canvas, timestamp=t, external_detections=detections)

    # --- Per-frame diagnostics ---
    if fi % 60 == 0:
        match_info = pipeline.frame_tracker._match_info
        H = pipeline.frame_tracker.H_calib_current
        l1_diff = float(np.sum(np.abs(H - np.eye(3))))

        n_in = compute_in_bounds(
            H,
            pipeline.projector._calib_world_pts,
            pipeline._calib_rvec if hasattr(pipeline, '_calib_rvec') else None,
            pipeline._calib_tvec if hasattr(pipeline, '_calib_tvec') else None,
            K, np.zeros((4, 1)), (1920, 1080))

        # Get current active athlete info
        active_lanes = sorted(pipeline.assigner.athletes.keys())
        confs = {l: pipeline.assigner.athletes[l].tracking_confidence for l in active_lanes}
        dms = {l: pipeline.assigner.athletes[l].d_m for l in active_lanes}
        coast = {l: pipeline.assigner.athletes[l].coast_count for l in active_lanes}
        n_finish = len(pipeline.standings.finish_times)

        print(f"[frame {fi:04d}] L1(H-I)={l1_diff:.4f}  in_bounds={n_in}  "
              f"n_active={len(active_lanes)}  n_fin={n_finish}")
        print(f"  match_info={match_info}")
        print(f"  confs={ {l: round(confs[l], 3) for l in active_lanes} }")
        print(f"  dms={ {l: round(dms[l], 2) for l in active_lanes} }")
        print(f"  coast={ {l: coast[l] for l in active_lanes if coast[l] > 0} }")

        if l1_diff > 2.0:
            print(f"  *** LARGE H L1 DIFF ***")
        if n_in < 10 and len(active_lanes) > 0:
            print(f"  *** VERY FEW IN-BOUNDS PnP POINTS ***")

    # Track athletes that drop to zero confidence
    for lane in range(1, 9):
        a = pipeline.assigner.athletes.get(lane)
        if a is not None:
            conf = a.tracking_confidence
            if lane not in dropped_log and last_conf.get(lane, 0) > 1e-6 and conf <= 1e-6:
                print(f"[frame {fi:04d}] CONF_DROP lane {lane}: d_m={a.d_m:.2f} d_m_min={a.dm_min:.2f} d_m_max={a.dm_max:.2f} coast={a.coast_count} frames_missed={a.frames_missed}")
                dropped_log.add(lane)
            last_conf[lane] = conf
        else:
            if lane not in removed_log:
                print(f"[frame {fi:04d}] REMOVED lane {lane}")
                removed_log.add(lane)

    # Track finishes
    for lane in range(1, 9):
        if lane in pipeline.standings.finish_times and lane not in finish_log:
            print(f"[frame {fi:04d}] FINISHED lane {lane} at {pipeline.standings.finish_times[lane]:.3f}s")
            finish_log.add(lane)

    if pipeline.timer.race_finished:
        print(f"[frame {fi:04d}] Race finished (all 8 done or forced stop)")
        break

n_fin = len(pipeline.standings.finish_times)
print(f"\n{'='*60}")
print("FINAL DIAGNOSTIC REPORT")
print(f"{'='*60}")
print(f"Frames processed: {fi+1}")
print(f"Finished athletes: {n_fin}/8")
print(f"Timer started: {pipeline.timer.race_started}")
print(f"Timer finished: {pipeline.timer.race_finished}")

for lane in range(1, 9):
    ft = pipeline.standings.finish_times.get(lane)
    if ft is not None:
        print(f"  Lane {lane}: FINISHED at {ft:.3f}s")
    else:
        a = pipeline.assigner.athletes.get(lane)
        if a is not None:
            print(f"  Lane {lane}: d_m={a.d_m:.2f} conf={a.tracking_confidence:.4f} coast={a.coast_count} frames_missed={a.frames_missed}")
        else:
            print(f"  Lane {lane}: REMOVED from assigner")

print(f"\nDropped lanes (confidence→0): {sorted(dropped_log)}")
print(f"Removed lanes: {sorted(removed_log)}")
print(f"Finished lanes: {sorted(finish_log)}")

expected_frame = int(race_len / SPEED * fps)
frame_error = abs(fi - expected_frame)
print(f"\nExpected finish at frame ~{expected_frame} ({expected_frame/fps:.1f}s)")
print(f"Actual finish at frame {fi+1} ({fi/fps:.1f}s)")
print(f"Frame error: {frame_error} ({frame_error/fps:.1f}s)")

if not pipeline.timer.race_finished:
    print("FAILURE: timer not stopped")
elif n_fin < 8:
    print(f"FAILURE: only {n_fin}/8 finished")
else:
    max_frame_err = int(0.5 * fps)
    if frame_error > max_frame_err:
        print(f"FAILURE: finish time error {frame_error/fps:.1f}s > 0.5s limit")
    else:
        print(f"PASS")
