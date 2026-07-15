import numpy as np
import cv2
import time
from pathlib import Path
from typing import Optional

from calibration.coords import TrackGeometry, ImageCoord
from calibration.projector import Projector
from calibration.calibrator import Calibrator
from calibration.lane_tracker import LaneFeatureTracker
from detection.detector import BaseDetector, DummyDetector
from tracking.lane_assigner import LaneAssigner
from tracking.position_estimator import PositionEstimator
from rendering.occlusion_guard import OcclusionGuard
from rendering.graphic_factory import GraphicContent
from rendering.decal_renderer import DecalRenderer, DecalInstance
from rendering.debug_overlay import DebugOverlay
from pipeline.ranking import RankingCalculator
from pipeline.timing import RaceTimer
from pipeline.preprocessor import Preprocessor
from pipeline.smoother import PositionSmoother
from pipeline.edge_cases import EdgeCaseDetector
from pipeline.dynamic_camera import DynamicCamera
from rendering.standings import StandingsPanel


class TrackARPipeline:
    def __init__(self, camera_matrix: np.ndarray | None = None, geometry: TrackGeometry | None = None):
        self.geometry = geometry if geometry is not None else TrackGeometry()
        img_size = (1920, 1080)
        self.calibrator = Calibrator(camera_matrix=camera_matrix, image_size=img_size)
        self.projector = Projector(
            self.calibrator.camera_matrix,
            self.calibrator.dist_coeffs,
        )
        self.detector: BaseDetector = DummyDetector(num_athletes=8)
        self.assigner = LaneAssigner(self.geometry, self.projector)
        self.estimator = PositionEstimator(self.geometry, self.projector)
        self.occlusion_guard = OcclusionGuard(self.geometry, self.projector)
        self.decal_renderer = DecalRenderer(self.projector)
        self.debug_overlay = DebugOverlay()
        self.ranking = RankingCalculator()
        self.timer = RaceTimer()
        self.preprocessor = Preprocessor()
        self.smoother = PositionSmoother()
        self.edge_detector = EdgeCaseDetector(self.geometry)
        self.dynamic_camera = DynamicCamera(self.projector)
        self.standings = StandingsPanel()
        self.frame_tracker = LaneFeatureTracker(max_features=600, quality_level=0.005, min_distance=3.0)
        self.calibrated = False
        self.running = False
        self.frame_count = 0
        self.fps = 0.0
        self.fps_history: list[float] = []
        self.athlete_names: dict[int, str] = {}
        self._texture_cache: dict[int, tuple[str, np.ndarray]] = {}
        self._anchor_timestamps: dict[int, float] = {}
        self._copy_buffer: np.ndarray | None = None
        self._start_decisions: list[int] = []  # rolling window for race start

    def calibrate_from_points(self, world_pts, image_pts):
        self.calibrator.solve_pnp(world_pts, image_pts)
        self.projector.set_extrinsics(self.calibrator.rvec, self.calibrator.tvec)
        self.projector.set_calibration_world_pts(world_pts)
        self._calib_rvec = self.calibrator.rvec.copy()
        self._calib_tvec = self.calibrator.tvec.copy()
        self.calibrated = True
        self._compute_track_bbox()

    def _compute_track_bbox(self):
        """Compute track bounding box in image space for audience filtering."""
        if self.geometry._model is not None:
            return
        margin = 40
        pts = []
        for lane in (1, 8):
            for dm in (0.0, self.geometry.length):
                wc = self.geometry.world_coord(lane, dm)
                ic = self.projector.project(wc)
                pts.append((ic.u, ic.v))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)
        self.assigner.set_track_bbox_100m(bbox)

    def set_detector(self, detector: BaseDetector):
        self.detector = detector

    def set_athlete_name(self, lane: int, name: str):
        self.athlete_names[lane] = name

    def reset(self):
        self.assigner = LaneAssigner(self.geometry, self.projector)
        self.estimator = PositionEstimator(self.geometry, self.projector)
        self.ranking = RankingCalculator()
        self.timer = RaceTimer()
        self.smoother = PositionSmoother()
        self.standings.reset()
        self.frame_count = 0
        self.fps_history.clear()

    def process_frame(self, frame: np.ndarray, timestamp: float | None = None,
                      external_detections: list | None = None,
                      frame_dt: float | None = None) -> np.ndarray:
        if timestamp is None:
            timestamp = time.time()
        if not self.calibrated:
            return frame
        self.frame_count += 1
        t_start = time.perf_counter()
        if frame_dt is None:
            frame_dt = 1.0 / max(self.fps, 1.0)
        preprocessed = self.preprocessor.process(frame, timestamp)
        # Update frame tracker: compute homography from calibration frame to current
        if self.frame_tracker.need_update():
            gray = preprocessed.original if preprocessed.original.ndim == 2 else cv2.cvtColor(preprocessed.original, cv2.COLOR_BGR2GRAY)
            self.frame_tracker.update(gray)
            H = self.frame_tracker.H_calib_current
            self.assigner.set_H_calib_current(H)
            # Project calibration world points through the ORIGINAL calibration
            # extrinsics, warp through cumulative H_calib_current, re-solve PnP.
            # Using the same reference every frame avoids drift accumulation.
            saved_r, saved_t = self.projector.rvec, self.projector.tvec
            if hasattr(self, '_calib_rvec') and self._calib_rvec is not None:
                self.projector.rvec = self._calib_rvec
                self.projector.tvec = self._calib_tvec
            self.projector.track_homography(H)
            # project() in the saved_r/saved_t was never tracked to current frame --
            # but track_homography just updated projector to the current frame's pose
            # After extrinsics update, projector is synced to current frame
            self.assigner.set_H_calib_current(np.eye(3, dtype=np.float64))
        if not self.frame_tracker.is_ready():
            self.assigner.set_H_calib_current(None)

        if external_detections is not None:
            detections = external_detections
        else:
            detections = self.detector.detect(preprocessed.original)
        athletes = self.assigner.process_frame(detections, frame_dt=frame_dt)
        positions = self.estimator.estimate(athletes, timestamp)
        if not self.timer.race_started:
            past_threshold = sum(1 for p in positions if p.d_m > 1.0 and p.speed_mps > 2.0 and p.confidence > 0.0)
            self._start_decisions.append(past_threshold)
            if len(self._start_decisions) > 3:
                self._start_decisions.pop(0)
            consistent = sum(1 for v in self._start_decisions if v >= 3) >= 2
            if consistent:
                self.timer.start_race(timestamp)
        current_time = self.timer.get_elapsed(timestamp)
        # Always include all 8 lanes in positions so standings shows every lane
        tracked_lanes = {p.lane for p in positions}
        for lane in range(1, 9):
            if lane not in tracked_lanes:
                from tracking.position_estimator import AthletePosition
                positions.append(AthletePosition(
                    lane=lane, athlete_id=lane, d_m=0.0, y_world=0.0,
                    speed_mps=0.0, timestamp=timestamp,
                    frame_count=self.frame_count, confidence=0.0,
                ))
        ranks = self.ranking.compute(positions, current_time)
        if self.dynamic_camera.follow_mode:
            self.dynamic_camera.update(positions)
        all_athletes_list = list(athletes.values())
        anchors = {}
        for lane, athlete in athletes.items():
            distance_to_end = self.geometry.length - athlete.d_m
            anchor = self.occlusion_guard.compute_safe_position(
                athlete, all_athletes_list, distance_to_end
            )
            pos = next((p for p in positions if p.lane == lane), None)
            conf = pos.confidence if pos else 1.0
            smoothed = self.smoother.smooth_anchor(lane, anchor, conf)
            # Transform anchor image coords from calibration frame to current frame via homography
            if self.frame_tracker is not None:
                u_c, v_c = self.frame_tracker.calib_to_current(smoothed.image.u, smoothed.image.v)
                smoothed.image = ImageCoord(u_c, v_c)
            anchors[lane] = smoothed
        alerts = self.edge_detector.check_all(athletes, positions)
        output = preprocessed.original
        for rank_entry in ranks:
            lane = rank_entry.lane
            if lane not in athletes or lane not in anchors:
                continue
            if rank_entry.confidence <= 0.0:
                continue
            anchor = anchors[lane]
            name = self.athlete_names.get(lane, f"Athlete {rank_entry.athlete_id}")
            time_str = self.timer.format_time(rank_entry.time)
            cache_key = f"{rank_entry.rank}|{time_str}|{name}|{lane}"
            cached = self._texture_cache.get(lane)
            if cached and cached[0] == cache_key:
                texture = cached[1]
            else:
                graphic = GraphicContent(rank=rank_entry.rank, time_str=time_str, name=name, lane=lane)
                texture = graphic.render_texture()
                self._texture_cache[lane] = (cache_key, texture)
            self.decal_renderer.render_decal(output, DecalInstance(None, anchor, texture))
        self.debug_overlay.draw(output, athletes, anchors, self.frame_count, self.fps)
        finish_distances = {lane: self.geometry.finish_distance(lane) for lane in range(1, 9)}
        self.standings.draw(output, ranks, positions, self.timer, self.athlete_names, self.geometry.length, timestamp, finish_distances)
        # Stop the timer once all athletes have finished
        all_finished = len(self.standings.finish_times) >= 8
        if all_finished and not self.timer.race_finished:
            self.timer.finish_race(timestamp)
        t_end = time.perf_counter()
        frame_time_ms = (t_end - t_start) * 1000
        self.fps_history.append(1000.0 / max(frame_time_ms, 1))
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        self.fps = sum(self.fps_history) / max(len(self.fps_history), 1)
        return output

    def run_on_video(self, video_path: str, output_path: str | None = None, max_frames: int = -1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames > 0 and frame_idx > max_frames:
                break
            timestamp = frame_idx / fps
            output = self.process_frame(frame, timestamp, frame_dt=1.0/fps)
            if writer:
                writer.write(output)
            cv2.imshow("TrackAR Pipeline", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    def run_live(self, camera_id: int = 0):
        import time as _time
        prev_t = _time.time()
        cap = cv2.VideoCapture(camera_id)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            now = _time.time()
            dt = now - prev_t
            prev_t = now
            output = self.process_frame(frame, frame_dt=dt)
            cv2.imshow("TrackAR Live", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        cap.release()
        cv2.destroyAllWindows()
