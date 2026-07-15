import sys
import numpy as np
import cv2
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from calibration.calibrator import Calibrator
from detection.detector import DummyDetector
from pipeline.main_pipeline import TrackARPipeline


def generate_test_video():
    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    video_path = str(out_dir / "integration_test_src.mp4")
    fps, w, h = 60, 1920, 1080
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (w, h))
    geom = TrackGeometry()
    lane_h = h / 8
    speeds = [9.5, 9.8, 10.2, 9.0, 8.5, 9.3, 8.8, 10.5]
    total_frames = 600
    for f_idx in range(total_frames):
        t = f_idx / fps
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 35
        for i in range(8):
            y0 = int(i * lane_h)
            cv2.rectangle(canvas, (0, y0), (w, y0 + int(lane_h)), (55, 55, 55), 1)
        cv2.line(canvas, (120, 0), (120, h), (0, 0, 255), 3)
        cv2.putText(canvas, "START", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        fx_px = int(w * 0.85)
        cv2.line(canvas, (fx_px, 0), (fx_px, h), (0, 255, 0), 3)
        cv2.putText(canvas, "FINISH", (fx_px + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for lane in range(8):
            d_m = max(0, min(t * speeds[lane], 100))
            cx = int(120 + (d_m / 100.0) * (fx_px - 120))
            cy = int((lane + 0.5) * lane_h)
            color = (int(50 + lane * 25) % 255, int(100 + lane * 15) % 255, int(200 - lane * 20) % 255)
            cv2.circle(canvas, (cx, cy), 15, color, -1)
            cv2.rectangle(canvas, (cx - 12, cy - 25), (cx + 12, cy + 25), color, 2)
            cv2.putText(canvas, f"L{lane+1}", (cx - 35, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(canvas)
    writer.release()
    print(f"Test video generated: {video_path} ({total_frames} frames)")
    return video_path


def main():
    print("TrackAR Integration Test (P4)")
    print("=" * 50)
    video_path = generate_test_video()
    print("\nInitializing pipeline...")
    pipeline = TrackARPipeline()
    K = np.array([[800, 0, 960], [0, 800, 540], [0, 0, 1]], dtype=np.float64)
    geom = TrackGeometry()
    pipeline = TrackARPipeline(camera_matrix=K)
    world_img_pairs = [
        (WorldCoord(0.0, geom.lane_center_y(1)), ImageCoord(1161, 802)),
        (WorldCoord(0.0, geom.lane_center_y(8)), ImageCoord(1311, 775)),
        (WorldCoord(100.0, geom.lane_center_y(1)), ImageCoord(851, 467)),
        (WorldCoord(100.0, geom.lane_center_y(8)), ImageCoord(902, 465)),
    ]
    w_pts = [p[0] for p in world_img_pairs]
    i_pts = [p[1] for p in world_img_pairs]
    pipeline.calibrate_from_points(w_pts, i_pts)
    pipeline.detector = DummyDetector(num_athletes=8)
    for lane in range(1, 9):
        pipeline.set_athlete_name(lane, f"Athlete {lane}")
    print("Pipeline calibrated and ready.")
    print(f"Processing video (simulated 10s @ 60fps)...")
    t_start = time.time()
    pipeline.run_on_video(video_path, max_frames=300)
    t_elapsed = time.time() - t_start
    print(f"\nIntegration test complete.")
    print(f"Processed 300 frames in {t_elapsed:.2f}s ({300/t_elapsed:.1f} effective fps)")
    print(f"Check output window for visual results.")


if __name__ == "__main__":
    main()
