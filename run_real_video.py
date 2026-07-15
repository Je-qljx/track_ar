"""
Run TrackAR on a real race video with click-based calibration.

Usage:
    python run_real_video.py race_video.mp4
    python run_real_video.py race_video.mp4 --fx 2000 --no-yolo
    python run_real_video.py race_video.mp4 --output result.mp4 --max-frames 300
"""
import sys
import numpy as np
import cv2
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.frame_tracker import FrameTracker
from pipeline.main_pipeline import TrackARPipeline

CALIB_NAMES = ["Start x Lane1", "Start x Lane8", "Finish x Lane1", "Finish x Lane8"]
CALIB_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (0, 255, 255)]


class ClickCalibrator:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.frame: np.ndarray | None = None
        self.points: list[tuple[int, int]] = []
        self.frames: list[np.ndarray] = []
        self.current = 0
        self._seek(0)

    def _seek(self, idx: int):
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, self.frame = self.cap.read()
        if not ret:
            return False
        self.frame_idx = idx
        return True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current < 4:
            self.points.append((x, y))
            self.frames.append(self.frame.copy())
            self.current += 1
            self._redraw()


    def _redraw(self):
        self.display = self.frame.copy()
        for i, (px, py) in enumerate(self.points):
            cv2.circle(self.display, (px, py), 8, CALIB_COLORS[i], -1)
            cv2.putText(self.display, f"{i+1}", (px + 10, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CALIB_COLORS[i], 2)
            cv2.putText(self.display, CALIB_NAMES[i], (px + 10, py + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CALIB_COLORS[i], 1)
        y_offset = 60
        if self.current < 4:
            remaining = [f"  Click {i+1}: {CALIB_NAMES[i]}" for i in range(self.current, 4)]
            for li, line in enumerate(remaining):
                cv2.putText(self.display, line, (30, y_offset + li * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += len(remaining) * 30 + 10
        if self.points:
            cv2.putText(self.display, "Points persist across frames", (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_offset += 22
        nav_info = f"Frame {self.frame_idx + 1}/{self.total_frames}" + (
            f"  (clicked: {', '.join(f'#{i+1}' for i in range(len(self.points)))})" if self.points else "")
        cv2.putText(self.display, nav_info, (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(self.display, "<-/-> = frame  [ = -10  ] = +10  SPACE=confirm  r=redo  q=quit",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    def run(self) -> tuple[list[tuple[int, int]], list[np.ndarray]] | None:
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        self._redraw()
        while True:
            cv2.imshow("Calibration", self.display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' ') and self.current == 4:
                break
            if key == ord('r'):
                self.points.clear()
                self.frames.clear()
                self.current = 0
                self._redraw()
            if key == ord('q'):
                cv2.destroyWindow("Calibration")
                return None
            if key == 81:  # left arrow
                self._seek(self.frame_idx - 1)
                self._redraw()
            if key == 83:  # right arrow
                self._seek(self.frame_idx + 1)
                self._redraw()
            if key == ord('['):
                self._seek(self.frame_idx - 10)
                self._redraw()
            if key == ord(']'):
                self._seek(self.frame_idx + 10)
                self._redraw()
        cv2.destroyWindow("Calibration")
        return self.points, self.frames


def main():
    parser = argparse.ArgumentParser(description="TrackAR - Real Video Runner with Click Calibration")
    parser.add_argument("video", help="Path to the race video file")
    parser.add_argument("--output", "-o", default=None, help="Output video path (default: input_overlay.mp4)")
    parser.add_argument("--focal-mm", type=float, default=None, help="Lens focal length in 35mm equivalent mm (e.g. 200)")
    parser.add_argument("--fx", type=float, default=None, help="Focal length in pixels (overrides --focal-mm)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Max frames to process (-1 = all)")
    parser.add_argument("--no-yolo", action="store_true", help="Use dummy detector instead of YOLO")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path (default: yolov8s.pt)")
    parser.add_argument("--track-type", default="100m", choices=["100m", "400m"],
                        help="Track geometry type (default: 100m). Use 400m for oval tracks.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    out_path = args.output or str(video_path.with_name(video_path.stem + "_overlay" + video_path.suffix))

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.2f}fps, {total_frames} frames")

    # Camera matrix — user can provide 35mm equivalent focal length or raw pixel fx
    if args.fx is not None:
        fx = args.fx
        src = "--fx"
    elif args.focal_mm is not None:
        fx = args.focal_mm * w / 36.0
        src = f"--focal-mm {args.focal_mm}mm -> {fx:.0f} px"
    else:
        fx = 200.0 * w / 36.0  # sensible default: 200mm equiv
        src = f"default 200mm equiv -> {fx:.0f} px"
    K = np.array([[fx, 0, w / 2],
                  [0, fx, h / 2],
                  [0, 0, 1]], dtype=np.float64)
    print(f"[INFO] Focal length: {src}")

    print("\n" + "=" * 60)
    print("  CALIBRATION — Navigate to a frame showing BOTH start and finish lines,")
    print("  then click the 4 points in order:")
    print("    1. Start line x Lane 1 (bottom lane)")
    print("    2. Start line x Lane 8 (top lane)")
    print("    3. Finish line x Lane 1")
    print("    4. Finish line x Lane 8")
    print("  <- / ->  = step 1 frame    [ / ]  = step 10 frames")
    print("  Press SPACE to confirm, 'r' to redo, 'q' to quit")
    print("=" * 60)

    calibrator = ClickCalibrator(cap)
    result = calibrator.run()
    if result is None:
        print("[INFO] Calibration cancelled.")
        cap.release()
        return
    calib_pixels, calib_frames = result

    # Rectify calibration points: transform all to first click's frame (reference frame)
    print("\n[RECTIFY] Aligning all calibration points to reference frame (click 1's frame)...")
    ref_gray = cv2.cvtColor(calib_frames[0], cv2.COLOR_BGR2GRAY)
    # Use higher feature count for calibration (frames may be very different)
    ft = FrameTracker(max_width=640, skip_interval=1)
    ft.orb = cv2.ORB.create(nfeatures=2000)
    ft.set_reference(ref_gray)
    print(f"  Reference: {len(ft.ref_kp)} features detected")
    rectified_pixels = [calib_pixels[0]]  # first point is already in ref frame
    for i in range(1, 4):
        gray = cv2.cvtColor(calib_frames[i], cv2.COLOR_BGR2GRAY)
        H = ft.update(gray)
        u, v = ft.current_to_calib(float(calib_pixels[i][0]), float(calib_pixels[i][1]))
        rectified_pixels.append((int(round(u)), int(round(v))))
        print(f"  Point {i+1} -> ref frame: ({calib_pixels[i][0]}, {calib_pixels[i][1]}) -> ({int(round(u))}, {int(round(v))}), matches={len(ft._last_matches) if hasattr(ft, '_last_matches') else '?'}")
    print("[OK] Calibration rectified to reference frame (click 1's frame):")
    for i, (orig, rect) in enumerate(zip(calib_pixels, rectified_pixels)):
        print(f"  Point {i+1}: original ({orig[0]}, {orig[1]}) -> rectified ({rect[0]}, {rect[1]})")

    geom = TrackGeometry(track_type=args.track_type)
    world_pts = geom.calibration_world_points()
    image_pts = [ImageCoord(float(u), float(v)) for u, v in rectified_pixels]

    names = ["Start x Lane1", "Start x Lane8", "Finish x Lane1", "Finish x Lane8"]
    for name, wp in zip(names, world_pts):
        print(f"  {name}: world=({wp.x:.2f}, {wp.y:.2f}, {wp.z:.1f})")

    # Build pipeline
    pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)
    pipeline.calibrate_from_points(world_pts, image_pts)
    # Set reference frame on pipeline's frame_tracker
    pipeline.frame_tracker.set_reference(ref_gray)
    cal_err = pipeline.calibrator.get_projection_error(world_pts, image_pts)
    pipeline.calibrator.print_calibration_debug(world_pts, image_pts)
    if cal_err > 15:
        print(f"[WARN] Calibration error: {cal_err:.3f} px — 过高！正常应 < 5px")
        print("       可能原因：")
        print("        1. 点击位置不精确（未对准起终点线与车道中心线交点）")
        print("        2. 焦距参数 --focal-mm 与实际拍摄焦距不符（可尝试不同值试误差最小的）")
        print("        3. 起终点之间的实际距离不是 100m（视频为 400m 项目？需修改 world_pts）")
        print("        4. 视频经过裁切/数码变焦，实际焦距与标注不符")
    else:
        print(f"[OK] Calibration error: {cal_err:.3f} px — 正常")

    if args.no_yolo:
        print("[INFO] Using DummyDetector (no real detections)")
    else:
        try:
            from detection.detector import YOLODetector
            detector = YOLODetector(model_path=args.model)
            pipeline.set_detector(detector)
            print(f"[OK] YOLO detector loaded: {args.model}")
        except Exception as e:
            print(f"[WARN] Could not load YOLO ({e}). Falling back to DummyDetector.")

    for lane in range(1, 9):
        pipeline.set_athlete_name(lane, f"Athlete {lane}")

    # Reset to first frame for processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process video
    print(f"\n[OK] Processing {total_frames} frames -> {out_path}")
    pipeline.running = True
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (w, h))
    frame_idx = 0
    last_report = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if args.max_frames > 0 and frame_idx > args.max_frames:
            break

        timestamp = frame_idx / fps
        output = pipeline.process_frame(frame, timestamp, frame_dt=1.0/fps)

        cv2.imshow("TrackAR - Real Video", output)
        writer.write(output)

        # Progress every 10%
        pct = frame_idx / min(args.max_frames, total_frames) * 100 if args.max_frames > 0 else frame_idx / total_frames * 100
        if pct - last_report >= 10:
            print(f"  Progress: {pct:.0f}% ({frame_idx}/{min(args.max_frames, total_frames) if args.max_frames > 0 else total_frames})  FPS: {pipeline.fps:.1f}")
            last_report = pct

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Output saved to: {out_path}")
    print(f"       Processed {frame_idx} frames")


if __name__ == "__main__":
    main()
