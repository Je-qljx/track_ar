import numpy as np
import cv2
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from calibration.calibrator import Calibrator
from detection.detector import DummyDetector
from tracking.lane_assigner import LaneAssigner
from tracking.position_estimator import PositionEstimator


def setup_simulated_system():
    img_w, img_h = 1920, 1080
    fx, fy = 2400, 2400
    cx, cy = img_w / 2, img_h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    geom = TrackGeometry()
    world_img_pairs = [
        (WorldCoord(0.0, geom.lane_center_y(1)), ImageCoord(360, 920)),
        (WorldCoord(0.0, geom.lane_center_y(8)), ImageCoord(160, 220)),
        (WorldCoord(100.0, geom.lane_center_y(1)), ImageCoord(1550, 840)),
        (WorldCoord(100.0, geom.lane_center_y(8)), ImageCoord(1710, 140)),
    ]
    w_pts = [p[0] for p in world_img_pairs]
    i_pts = [p[1] for p in world_img_pairs]
    cal = Calibrator(camera_matrix=K, image_size=(img_w, img_h))
    cal.solve_pnp(w_pts, i_pts)
    proj = Projector(K)
    proj.set_extrinsics(cal.rvec, cal.tvec)
    return geom, proj, cal


def simulate_frame(detector, frame_idx: int):
    h, w = 1080, 1920
    lane_h = h / 8
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 30
    detections = []
    speeds = [9.5, 9.8, 10.2, 9.0, 8.5, 9.3, 8.8, 10.5]
    for i in range(8):
        d_m = min(frame_idx * 0.5 * speeds[i] / 60, 100.0)
        cy = int((i + 0.5) * lane_h)
        lane_w_px = 60
        cx = int(w * 0.15 + (d_m / 100.0) * w * 0.7)
        x1 = max(0, cx - lane_w_px // 2)
        y1 = max(0, cy - 60)
        x2 = min(w, cx + lane_w_px // 2)
        y2 = min(h, cy + 60)
        detections.append(type("Det", (), {
            "bbox": (x1, y1, x2, y2),
            "confidence": 0.95,
            "bottom_center": ((x1 + x2) / 2, y2),
        })())
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"L{i+1} {d_m:.1f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas, detections


def main():
    geom, proj, cal = setup_simulated_system()
    detector = DummyDetector(num_athletes=8)
    assigner = LaneAssigner(geom, proj)
    estimator = PositionEstimator(geom, proj)
    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    writer = None
    total_frames = 150
    for frame_idx in range(total_frames):
        canvas, detections = simulate_frame(detector, frame_idx)
        athletes = assigner.process_frame(detections)
        positions = estimator.estimate(athletes, timestamp=frame_idx / 60.0)
        for pos in positions:
            lane = pos.lane
            wc = WorldCoord(pos.d_m, geom.lane_center_y(lane), 0.0)
            ip = proj.project(wc)
            cv2.circle(canvas, (int(ip.u), int(ip.v)), 8, (0, 0, 255), -1)
            cv2.putText(canvas, f"#{pos.athlete_id} L{lane} {pos.d_m:.1f}m {pos.speed_mps:.1f}m/s",
                        (int(ip.u) + 15, int(ip.v)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos = 30
        for pos in sorted(positions, key=lambda p: p.d_m, reverse=True):
            cv2.putText(canvas, f"L{pos.lane} #{pos.athlete_id}: {pos.d_m:.1f}m",
                        (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_pos += 25
        if writer is None:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_dir / "p2_tracking_test.mp4"), fourcc, 60.0, (1920, 1080))
        writer.write(canvas)
        cv2.imshow("P2 Tracking", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Tracking video saved to {out_dir / 'p2_tracking_test.mp4'}")
    print(f"Processed {total_frames} frames, tracked {len(positions)} athletes")


if __name__ == "__main__":
    main()
