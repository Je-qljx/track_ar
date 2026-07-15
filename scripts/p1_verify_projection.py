import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.coords import WorldCoord, ImageCoord, TrackGeometry
from calibration.projector import Projector
from calibration.calibrator import Calibrator


def make_simulated_camera():
    img_w, img_h = 1920, 1080
    fx, fy = 2400, 2400
    cx, cy = img_w / 2, img_h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    geom = TrackGeometry()
    world_pts = [w for _, w in geom.reference_points_world()]
    rvec = np.array([[0.5], [-0.3], [0.1]], dtype=np.float64)
    tvec = np.array([[0], [-15], [20]], dtype=np.float64)
    img_pts = []
    for w in world_pts:
        pts_3d = w.as_array.reshape(1, 1, 3).astype(np.float64)
        proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
        img_pts.append(ImageCoord(u=float(proj[0, 0, 0]), v=float(proj[0, 0, 1])))
    cal = Calibrator(camera_matrix=K, image_size=(img_w, img_h))
    cal.solve_pnp(world_pts, img_pts)
    proj = Projector(K, dist)
    proj.set_extrinsics(cal.rvec, cal.tvec)
    err = cal.get_projection_error(world_pts, img_pts)
    print(f"Camera matrix:\n{K}")
    print(f"Rotation vector:\n{cal.rvec}")
    print(f"Translation vector:\n{cal.tvec}")
    print(f"Reprojection error: {err:.4f} pixels")
    for name, w in geom.reference_points_world():
        ip = proj.project(w)
        print(f"  {name}: World({w.x:.1f}, {w.y:.1f}) -> Image({ip.u:.1f}, {ip.v:.1f})")
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 50
    for _, w in geom.reference_points_world():
        ip = proj.project(w)
        cv2.circle(canvas, (int(ip.u), int(ip.v)), 8, (0, 255, 0), -1)
    athlete_positions = [
        WorldCoord(15.0, geom.lane_center_y(1), 0.0),
        WorldCoord(22.0, geom.lane_center_y(2), 0.0),
        WorldCoord(30.0, geom.lane_center_y(3), 0.0),
        WorldCoord(18.0, geom.lane_center_y(4), 0.0),
        WorldCoord(45.0, geom.lane_center_y(5), 0.0),
        WorldCoord(38.0, geom.lane_center_y(6), 0.0),
        WorldCoord(12.0, geom.lane_center_y(7), 0.0),
        WorldCoord(50.0, geom.lane_center_y(8), 0.0),
    ]
    for w in athlete_positions:
        ip = proj.project(w)
        lane = geom.lane_from_y(w.y)
        cv2.circle(canvas, (int(ip.u), int(ip.v)), 12, (0, 0, 255), -1)
        cv2.putText(canvas, f"L{lane}", (int(ip.u) + 15, int(ip.v)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # draw start/finish lines
    for x, label in [(0, "START"), (100, "FINISH")]:
        for lane in [1, 8]:
            w = WorldCoord(x, geom.lane_center_y(lane), 0.0)
            ip = proj.project(w)
            cv2.circle(canvas, (int(ip.u), int(ip.v)), 6, (255, 255, 0), -1)
    cv2.putText(canvas, "P1 Verification - Projection Test", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    out_path = Path(__file__).resolve().parent.parent / "output" / "p1_projection_test.png"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"\nOutput saved to {out_path}")
    return cal


if __name__ == "__main__":
    make_simulated_camera()
