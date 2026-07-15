import sys; sys.path.insert(0, 'D:/track_ar')
import numpy as np
import cv2
from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from pipeline.main_pipeline import TrackARPipeline
from tests.synthetic_scene import SyntheticScene

SPEED = 9.5
K = np.array([[700, 0, 960], [0, 700, 540], [0, 0, 1]], dtype=np.float64)

R0_SIDE = np.array([[0.0], [2.3805], [2.0501]], dtype=np.float64)
T0_SIDE = np.array([[50.0], [-0.02], [40.49]], dtype=np.float64)

def build_calibration_target(geom, dm_t, lane_t, w_t, h_t):
    cy = geom.lane_center_y(lane_t)
    world_pts = [
        WorldCoord(dm_t - w_t/2, cy - h_t/2, 0.0),
        WorldCoord(dm_t + w_t/2, cy - h_t/2, 0.0),
        WorldCoord(dm_t + w_t/2, cy + h_t/2, 0.0),
        WorldCoord(dm_t - w_t/2, cy + h_t/2, 0.0),
    ]
    return world_pts

geom = TrackGeometry(track_type="100m")
pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)

print("=== CALIBRATION ===")
world_pts = build_calibration_target(geom, 50, 5, 0.420, 0.297)
w_arr = np.array([w.as_array for w in world_pts], dtype=np.float64)
proj, _ = cv2.projectPoints(w_arr, R0_SIDE, T0_SIDE, K, np.zeros((4, 1)))
image_pts = [ImageCoord(float(p[0, 0]), float(p[0, 1])) for p in proj]
print(f"Target world pts:")
for i, w in enumerate(world_pts):
    print(f"  {i}: ({w.x:.3f}, {w.y:.3f}) -> img ({image_pts[i].u:.1f}, {image_pts[i].v:.1f})")

pipeline.calibrate_from_points(world_pts, image_pts)
cal_err = pipeline.calibrator.get_projection_error(world_pts, image_pts)
print(f"Calibration error: {cal_err:.3f}px")

pipeline.calibrator.print_calibration_debug(world_pts, image_pts)

# Verify: project track corners through calibrated extrinsics
print("\n=== TRACK CORNER PROJECTION (calibrated) ===")
for lane in (1, 8):
    for dm in (0.0, 100.0):
        wc = geom.world_coord(lane, dm)
        ic = pipeline.projector.project(wc)
        print(f"  Lane {lane}, dm={dm:.0f}: ({ic.u:.1f}, {ic.v:.1f})")

# Ground truth projector
render_proj = Projector(K, np.zeros((4, 1)))
render_proj.set_extrinsics(R0_SIDE.copy(), T0_SIDE.copy())

print("\n=== TRACK CORNER PROJECTION (ground truth) ===")
for lane in (1, 8):
    for dm in (0.0, 100.0):
        wc = geom.world_coord(lane, dm)
        ic = render_proj.project(wc)
        print(f"  Lane {lane}, dm={dm:.0f}: ({ic.u:.1f}, {ic.v:.1f})")

scene = SyntheticScene(render_proj, geom, speeds=[SPEED] * 8)

print("\n=== FRAME-BY-FRAME DEBUG ===")
for fi in range(30):
    t = fi / 60.0
    athletes = scene.update(t)
    canvas = scene.render(athletes)
    detections = scene.get_detections(athletes)
    pipeline.process_frame(canvas, timestamp=t, external_detections=detections)

    # Ground truth athlete positions in image
    gt_img_positions = {}
    for a in athletes:
        ip = scene._get_img_pos(a)
        gt_img_positions[a.lane] = (ip.u, ip.v)

    # Pipeline athlete states
    pipe_athletes = pipeline.assigner.athletes

    # Frame tracker info
    ft_info = pipeline.frame_tracker.last_match_info if hasattr(pipeline.frame_tracker, 'last_match_info') else {}

    print(f"\n--- Frame {fi} (t={t:.3f}s) ---")
    print(f"  Detections: {len(detections)}")
    for i, d in enumerate(detections):
        u, v = d.bottom_center
        print(f"    Det {i}: bottom_center=({u:.1f}, {v:.1f}) conf={d.confidence:.2f}")

    print(f"  GT athlete img positions:")
    for lane, (u, v) in sorted(gt_img_positions.items()):
        print(f"    L{lane}: ({u:.1f}, {v:.1f})")

    print(f"  Pipeline athletes: {len(pipe_athletes)}")
    for lane in sorted(pipe_athletes.keys()):
        a = pipe_athletes[lane]
        # Predict pixel in current frame
        pu, pv = pipeline.assigner._predict_pixel_current(a)
        # Detection pixel if matched
        det_str = ""
        if a.detection is not None:
            du, dv = a.detection.bottom_center
            det_str = f" det=({du:.1f}, {dv:.1f}) px_dist={np.hypot(du-pu, dv-pv):.1f}"
        # GT pixel for comparison
        gt_u, gt_v = gt_img_positions.get(lane, (0, 0))
        gt_px_dist = np.hypot(gt_u - pu, gt_v - pv)
        print(f"    L{lane}: id={a.athlete_id} dm={a.d_m:.2f} speed={a.speed_mps:.2f} "
              f"conf={a.tracking_confidence:.3f} frames_missed={a.frames_missed} "
              f"coast={a.coast_count}{det_str}"
              f" gt_px_dist={gt_px_dist:.1f}")

    print(f"  FrameTracker H_calib_current:")
    H = pipeline.frame_tracker.H_calib_current
    print(f"    [[{H[0,0]:.4f}, {H[0,1]:.4f}, {H[0,2]:.1f}],")
    print(f"     [{H[1,0]:.4f}, {H[1,1]:.4f}, {H[1,2]:.1f}],")
    print(f"     [{H[2,0]:.4f}, {H[2,1]:.4f}, {H[2,2]:.1f}]]")
    print(f"    method: {ft_info.get('method', 'n/a')}, matches: {ft_info.get('first_matches', ft_info.get('pairwise_matches', 'n/a'))}")

    # Check: does track_homography change the pose?
    old_r = pipeline.projector.rvec.copy()
    old_t = pipeline.projector.tvec.copy()
    pipeline.projector.track_homography(H)
    r_diff = np.linalg.norm(pipeline.projector.rvec - old_r)
    t_diff = np.linalg.norm(pipeline.projector.tvec - old_t)
    print(f"    track_homography: r_diff={r_diff:.6f}, t_diff={t_diff:.6f}")

    # Check is_in_track_region for detections
    for i, d in enumerate(detections):
        u, v = d.bottom_center
        cu, cv = pipeline.assigner._current_to_calib(u, v)
        in_region = pipeline.assigner._is_in_track_region(cu, cv)
        print(f"    Det {i} in_track_region: {in_region} (calib=({cu:.1f},{cv:.1f}))")

    # Check unproject_to_ground consistency
    for a in athletes:
        wc = geom.world_coord(a.lane, a.d_m)
        ic = pipeline.projector.project(wc)
        world_back = pipeline.projector.unproject_to_ground(ic)
        dm_err = abs(world_back.x - a.d_m)
        if dm_err > 2.0:
            print(f"    *** UNPROJECT ERROR L{a.lane}: dm={a.d_m:.1f} -> project -> unproject -> {world_back.x:.1f} (err={dm_err:.1f}m)")

print("\n=== FINAL STATE ===")
active = [a for a in pipeline.assigner.athletes.values() if a.tracking_confidence > 0]
print(f"Active athletes (conf>0): {len(active)}/8")
for lane in sorted(pipeline.assigner.athletes.keys()):
    a = pipeline.assigner.athletes[lane]
    print(f"  L{lane}: dm={a.d_m:.2f} conf={a.tracking_confidence:.3f} frames_missed={a.frames_missed} coast={a.coast_count}")
