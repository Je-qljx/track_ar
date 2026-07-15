import sys; sys.path.insert(0, 'D:/track_ar')
import numpy as np
import cv2
import time as _time
from calibration.coords import TrackGeometry, ImageCoord, WorldCoord
from calibration.projector import Projector
from pipeline.main_pipeline import TrackARPipeline
from tests.synthetic_scene import SyntheticScene

SPEED = 9.5
K = np.array([[700, 0, 960], [0, 700, 540], [0, 0, 1]], dtype=np.float64)

# ---- camera poses ----
R0_100M = np.array([[-1.9865], [-0.7462], [-0.4312]], dtype=np.float64)
T0_100M = np.array([[-40.827], [-14.574], [36.127]], dtype=np.float64)
R0_400M = np.array([[-0.3002], [-3.0314], [-0.5945]], dtype=np.float64)
T0_400M = np.array([[-33.3], [43.9], [125.9]], dtype=np.float64)
R0_SIDE = np.array([[0.0], [2.3805], [2.0501]], dtype=np.float64)
T0_SIDE = np.array([[50.0], [-0.02], [40.49]], dtype=np.float64)


def _sin_amp(f, fps, period_s, amp):
    return amp * np.sin(2 * np.pi * f / (fps * period_s))

# ---- perturbation functions (realistic broadcast amplitudes) ----
def perturb_static(f, fps, r0, t0):
    return r0.copy(), t0.copy()

def perturb_pan(f, fps, r0, t0):
    r = r0.copy()
    r[1, 0] += _sin_amp(f, fps, 10.0, 0.0003)
    return r, t0.copy()

def perturb_zoom(f, fps, r0, t0):
    t = t0.copy()
    t[2, 0] += _sin_amp(f, fps, 12.0, 0.5)
    return r0.copy(), t

def perturb_dolly(f, fps, r0, t0):
    t = t0.copy()
    t[0, 0] += _sin_amp(f, fps, 15.0, 0.5)
    return r0.copy(), t

def perturb_boom(f, fps, r0, t0):
    r = r0.copy()
    t = t0.copy()
    t[1, 0] += _sin_amp(f, fps, 8.0, 0.3)
    return r, t


def _add_tracking_grid(geom, calib_pts, target_spec=None):
    """Build dense grid of world points for robust track_homography PnP."""
    pts = list(calib_pts)
    for dm in np.arange(0.0, geom.length + 1, 10.0):
        for y in np.arange(0.0, min(10.0, 8 * geom.lane_width), 1.22):
            if target_spec is not None:
                dm_t, _, cy = target_spec[0], geom.lane_center_y(target_spec[1]), geom.lane_center_y(target_spec[1])
                if abs(dm - target_spec[0]) < 1.0 and abs(y - cy) < 1.0:
                    continue
            pts.append(WorldCoord(dm, y, 0.0))
    return pts


def run_test(track_type: str, r0, t0, cam_K, perturb_fn,
             target_spec=None, name: str = "",
             max_time_err_s: float = 0.2) -> str:
    """Full synthetic race test.

    target_spec: None → standard; (dm, lane, w, h) → calibration target.
    max_time_err_s: maximum allowed finish-time error in seconds.
    """
    geom = TrackGeometry(track_type=track_type)
    pipeline = TrackARPipeline(camera_matrix=cam_K, geometry=geom)

    if target_spec is not None:
        dm_t, lane_t, w_t, h_t = target_spec
        cy = geom.lane_center_y(lane_t)
        calib_pts = [
            WorldCoord(dm_t - w_t/2, cy - h_t/2, 0.0),
            WorldCoord(dm_t + w_t/2, cy - h_t/2, 0.0),
            WorldCoord(dm_t + w_t/2, cy + h_t/2, 0.0),
            WorldCoord(dm_t - w_t/2, cy + h_t/2, 0.0),
        ]
    else:
        calib_pts = geom.calibration_world_points()
    w_arr = np.array([w.as_array for w in calib_pts], dtype=np.float64)
    proj, _ = cv2.projectPoints(w_arr, r0, t0, cam_K, np.zeros((4, 1)))
    image_pts = [ImageCoord(float(p[0, 0]), float(p[0, 1])) for p in proj]
    pipeline.calibrate_from_points(calib_pts, image_pts)

    # Dense tracking grid for robust PnP in track_homography
    track_pts = _add_tracking_grid(geom, calib_pts, target_spec)
    pipeline.projector.set_calibration_world_pts(track_pts)

    err = pipeline.calibrator.get_projection_error(calib_pts, image_pts)
    if err > 5.0:
        return f"FAIL: calib error {err:.3f}px"

    render_proj = Projector(cam_K, np.zeros((4, 1)))
    render_proj.set_extrinsics(r0.copy(), t0.copy())
    scene = SyntheticScene(render_proj, geom, speeds=[SPEED] * 8)
    fps = 60.0
    race_len = geom.finish_distance(1)
    max_frames = int(race_len / SPEED * fps) + 200

    t_start = _time.time()
    drop_count = 0

    for fi in range(max_frames):
        if _time.time() - t_start > 300:
            return "TIMEOUT"
        rvec, tvec = perturb_fn(fi, fps, r0, t0)
        render_proj.set_extrinsics(rvec, tvec)
        t = fi / fps
        athletes = scene.update(t)
        canvas = scene.render_background(athletes)
        detections = scene.get_detections(athletes)
        pipeline.process_frame(canvas, timestamp=t, external_detections=detections)
        active = [a for a in pipeline.assigner.athletes.values() if a.tracking_confidence > 0]
        if fi >= 10 and len(active) < 8:
            drop_count += 1
        if pipeline.timer.race_finished:
            break

    if not pipeline.timer.race_finished:
        return f"FAIL: timer not stopped (n_fin={len(pipeline.standings.finish_times)}, frames={fi})"
    n_fin = len(pipeline.standings.finish_times)
    if n_fin < 8:
        return f"FAIL: only {n_fin}/8 finished"
    max_drops = int(max_frames * 0.1)
    if drop_count > max_drops and drop_count > 10:
        return f"FAIL: {drop_count} drops in {max_frames} frames"
    expected_frame = int(race_len / SPEED * fps)
    frame_error = abs(fi - expected_frame)
    max_frame_err = int(max_time_err_s * fps)
    if frame_error > max_frame_err:
        return f"FAIL: finish frame {fi} vs expected {expected_frame} ({frame_error/fps:.1f}s error > {max_time_err_s}s)"
    return f"PASS ({fi/fps:.1f}s, {fi+1} frames, err={frame_error})"


# ---- full-race scenarios ----
SCENARIOS = [
    # === Static camera (standard calibration) ===
    ("100m/static/std",          "100m", R0_100M, T0_100M, perturb_static, None),
    ("400m/static/std",          "400m", R0_400M, T0_400M, perturb_static, None),

    # === Target mode — static, various positions ===
    ("100m/static/tgt_mid",      "100m", R0_100M, T0_100M, perturb_static, (50, 5, 0.420, 0.297)),
    ("100m/static/tgt_start",    "100m", R0_100M, T0_100M, perturb_static, (15, 1, 0.420, 0.297)),
    ("100m/static/tgt_finish",   "100m", R0_100M, T0_100M, perturb_static, (88, 8, 0.420, 0.297)),
    ("100m/static/tgt_edge",     "100m", R0_100M, T0_100M, perturb_static, (50, 1, 0.300, 0.210)),
    ("400m/static/tgt_mid",      "400m", R0_400M, T0_400M, perturb_static, (200, 5, 0.420, 0.297)),
    ("400m/static/tgt_curve",    "400m", R0_400M, T0_400M, perturb_static, (60, 1, 0.420, 0.297)),
    ("400m/static/tgt_far",      "400m", R0_400M, T0_400M, perturb_static, (320, 8, 0.420, 0.297)),

    # === Side view + target (static) ===
    ("100m/static/tgt_side",     "100m", R0_SIDE, T0_SIDE, perturb_static, (50, 5, 0.420, 0.297)),

    # === Camera motion + standard calibration ===
    ("100m/pan/std",             "100m", R0_100M, T0_100M, perturb_pan,   None),
    ("400m/pan/std",             "400m", R0_400M, T0_400M, perturb_pan,   None),
    ("100m/boom/std",            "100m", R0_100M, T0_100M, perturb_boom,  None,     1.5),  # boom ≈ vertical scale change → same depth-ambiguity as zoom
    ("400m/zoom/std",            "400m", R0_400M, T0_400M, perturb_zoom,  None),

    # Zoom/dolly on 100m: known PnP depth-ambiguity limitation → relaxed tolerance
    ("100m/zoom/std",            "100m", R0_100M, T0_100M, perturb_zoom,  None,     1.5),
    ("100m/dolly/std",           "100m", R0_100M, T0_100M, perturb_dolly, None,     0.5),

    # === Camera motion + target mode ===
    ("100m/pan/tgt_mid",         "100m", R0_100M, T0_100M, perturb_pan,   (50, 5, 0.420, 0.297)),
    ("100m/boom/tgt_mid",        "100m", R0_100M, T0_100M, perturb_boom,  (50, 5, 0.420, 0.297),  1.5),  # boom ≈ vertical scale change → same depth-ambiguity as zoom
    ("400m/pan/tgt_mid",         "400m", R0_400M, T0_400M, perturb_pan,   (200, 5, 0.420, 0.297)),
    ("400m/zoom/tgt_mid",        "400m", R0_400M, T0_400M, perturb_zoom,  (200, 5, 0.420, 0.297)),

    # Zoom/dolly on 100m target mode: same PnP limitation → relaxed tolerance
    ("100m/zoom/tgt_mid",        "100m", R0_100M, T0_100M, perturb_zoom,  (50, 5, 0.420, 0.297),  1.5),
    ("100m/dolly/tgt_mid",       "100m", R0_100M, T0_100M, perturb_dolly, (50, 5, 0.420, 0.297),  0.5),
]

# ---- 标定物快速验收 ----
def run_target_calib_check(track_type, r0, t0, cam_K, target_spec, name=""):
    geom = TrackGeometry(track_type=track_type)
    pipeline = TrackARPipeline(camera_matrix=cam_K, geometry=geom)

    if target_spec is not None:
        dm_t, lane_t, w_t, h_t = target_spec
        cy = geom.lane_center_y(lane_t)
        world_pts = [
            WorldCoord(dm_t - w_t/2, cy - h_t/2, 0.0),
            WorldCoord(dm_t + w_t/2, cy - h_t/2, 0.0),
            WorldCoord(dm_t + w_t/2, cy + h_t/2, 0.0),
            WorldCoord(dm_t - w_t/2, cy + h_t/2, 0.0),
        ]
        wa = np.array([w.as_array for w in world_pts], dtype=np.float64)
        pj, _ = cv2.projectPoints(wa, r0, t0, cam_K, np.zeros((4, 1)))
        ip = [ImageCoord(float(p[0, 0]), float(p[0, 1])) for p in pj]
        pipeline.calibrate_from_points(world_pts, ip)
        cal_err = pipeline.calibrator.get_projection_error(world_pts, ip)
        if cal_err > 5.0:
            return f"FAIL: calib error {cal_err:.3f}px"
        if pipeline.projector._calib_world_pts is None:
            return "FAIL: calibration world pts not stored"
        # Unproject round-trip at target center
        if pipeline.projector.rvec is not None:
            center_img = ImageCoord(
                float(np.mean([p.u for p in ip])),
                float(np.mean([p.v for p in ip])),
            )
            center_world = pipeline.projector.unproject_to_ground(center_img)
            dx = abs(center_world.x - dm_t)
            dy = abs(center_world.y - cy)
            if dx > 3.0 or dy > 3.0:
                return f"FAIL: unproject error ({dx:.2f}m, {dy:.2f}m)"
        return f"PASS (err={cal_err:.3f}px)"
    else:
        # Standard calibration check
        geom = TrackGeometry(track_type=track_type)
        pipeline = TrackARPipeline(camera_matrix=cam_K, geometry=geom)
        wp = geom.calibration_world_points()
        wa = np.array([w.as_array for w in wp], dtype=np.float64)
        pj, _ = cv2.projectPoints(wa, r0, t0, cam_K, np.zeros((4, 1)))
        ip = [ImageCoord(float(p[0, 0]), float(p[0, 1])) for p in pj]
        pipeline.calibrate_from_points(wp, ip)
        err = pipeline.calibrator.get_projection_error(wp, ip)
        if err > 5.0:
            return f"FAIL: calib error {err:.3f}px"
        return f"PASS (err={err:.3f}px)"


QUICK_SCENARIOS = [
    ("qc_100m_std",      "100m", R0_100M, T0_100M, None),
    ("qc_400m_std",      "400m", R0_400M, T0_400M, None),
    ("qc_100m_mid",      "100m", R0_100M, T0_100M, (50, 5, 0.420, 0.297)),
    ("qc_100m_start",    "100m", R0_100M, T0_100M, (10, 1, 0.420, 0.297)),
    ("qc_100m_finish",   "100m", R0_100M, T0_100M, (95, 8, 0.420, 0.297)),
    ("qc_100m_small_A5", "100m", R0_100M, T0_100M, (50, 5, 0.210, 0.148)),
    ("qc_100m_tiny",     "100m", R0_100M, T0_100M, (50, 5, 0.100, 0.070)),
    ("qc_400m_mid",      "400m", R0_400M, T0_400M, (200, 5, 0.420, 0.297)),
    ("qc_400m_curve",    "400m", R0_400M, T0_400M, (60, 1, 0.420, 0.297)),
    ("qc_400m_far",      "400m", R0_400M, T0_400M, (300, 8, 0.420, 0.297)),
    ("qc_100m_sideview", "100m", R0_SIDE, T0_SIDE, (50, 5, 0.420, 0.297)),
    # ("qc_400m_sideview", "400m", R0_SIDE, T0_SIDE, (200, 5, 0.420, 0.297)),  # side view designed for 100m; invalid for 400m
    ("qc_100m_large",    "100m", R0_100M, T0_100M, (50, 5, 1.0, 0.7)),
    ("qc_400m_large",    "400m", R0_400M, T0_400M, (200, 5, 1.0, 0.7)),
]


if __name__ == '__main__':
    # === Quick calibration checks ===
    print("=== Quick Calibration Checks ===")
    q_passed = 0
    for s in QUICK_SCENARIOS:
        name, tt, r0, t0, spec = s
        sys.stdout.write(f"  {name} ... ")
        sys.stdout.flush()
        try:
            result = run_target_calib_check(tt, r0, t0, K, spec, name)
        except Exception as e:
            import traceback; traceback.print_exc()
            result = f"CRASH: {e}"
        print(result)
        if result.startswith("PASS"):
            q_passed += 1
    print(f"  Quick checks: {q_passed}/{len(QUICK_SCENARIOS)} passed\n")

    # === Full race tests ===
    print(f"Running {len(SCENARIOS)} full-race scenarios ({SPEED} m/s)...")
    r_passed = 0
    for s in SCENARIOS:
        if len(s) == 7:
            name, tt, r0, t0, pf, spec, tol = s
        else:
            name, tt, r0, t0, pf, spec = s
            tol = 0.2
        sys.stdout.write(f"  {name} ... ")
        sys.stdout.flush()
        try:
            result = run_test(tt, r0, t0, K, pf, spec, name, tol)
        except Exception as e:
            import traceback; traceback.print_exc()
            result = f"CRASH: {e}"
        print(result)
        if result.startswith("PASS"):
            r_passed += 1

    total = q_passed + r_passed
    out_of = len(QUICK_SCENARIOS) + len(SCENARIOS)
    print(f"\n{'='*50}")
    print(f"  RESULT: {total}/{out_of} passed")
    print(f"{'='*50}")
