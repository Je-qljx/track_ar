import sys; sys.path.insert(0, 'D:/track_ar')
import numpy as np
import cv2
import time
from collections import defaultdict
from calibration.coords import TrackGeometry, ImageCoord, WorldCoord
from calibration.projector import Projector
from pipeline.main_pipeline import TrackARPipeline
from tests.synthetic_scene import SyntheticScene

SPEED = 9.5
K = np.array([[700, 0, 960], [0, 700, 540], [0, 0, 1]], dtype=np.float64)
R0_100M = np.array([[-1.9865], [-0.7462], [-0.4312]], dtype=np.float64)
T0_100M = np.array([[-40.827], [-14.574], [36.127]], dtype=np.float64)


# ── Monkey-patch process_frame with instrumentation ────────────────────────

_N_TIMERS = defaultdict(list)

def _t_segment(name):
    def deco(meth):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            r = meth(*args, **kwargs)
            _N_TIMERS[name].append(time.perf_counter() - t0)
            return r
        return wrapper
    return deco

# Apply to the actual pipeline object after construction
_orig_process_frame = TrackARPipeline.process_frame

def _instrumented_process_frame(self, frame, timestamp=None, external_detections=None, frame_dt=None):
    if timestamp is None:
        timestamp = time.time()
    if not self.calibrated:
        return frame
    self.frame_count += 1
    t_start = time.perf_counter()
    if frame_dt is None:
        frame_dt = 1.0 / max(self.fps, 1.0)

    # 1. Preprocessor
    t0 = time.perf_counter()
    preprocessed = self.preprocessor.process(frame, timestamp)
    _N_TIMERS['preprocess'].append(time.perf_counter() - t0)

    # 2. Camera tracking (frame_tracker + homography + PnP)
    t0 = time.perf_counter()
    if self.frame_tracker.need_update():
        gray = preprocessed.original if preprocessed.original.ndim == 2 else cv2.cvtColor(preprocessed.original, cv2.COLOR_BGR2GRAY)
        self.frame_tracker.update(gray)
        H = self.frame_tracker.H_calib_current
        self.assigner.set_H_calib_current(H)
        saved_r, saved_t = self.projector.rvec, self.projector.tvec
        if hasattr(self, '_calib_rvec') and self._calib_rvec is not None:
            self.projector.rvec = self._calib_rvec
            self.projector.tvec = self._calib_tvec
        self.projector.track_homography(H)
        self.assigner.set_H_calib_current(np.eye(3, dtype=np.float64))
    if not self.frame_tracker.is_ready():
        self.assigner.set_H_calib_current(None)
    _N_TIMERS['camera_tracking'].append(time.perf_counter() - t0)

    # 3. Detection
    t0 = time.perf_counter()
    if external_detections is not None:
        detections = external_detections
    else:
        detections = self.detector.detect(preprocessed.original)
    _N_TIMERS['detection'].append(time.perf_counter() - t0)

    # 4. Lane assignment
    t0 = time.perf_counter()
    athletes = self.assigner.process_frame(detections, frame_dt=frame_dt)
    _N_TIMERS['lane_assignment'].append(time.perf_counter() - t0)

    # 5. Position estimation
    t0 = time.perf_counter()
    positions = self.estimator.estimate(athletes, timestamp)
    _N_TIMERS['position_estimation'].append(time.perf_counter() - t0)

    # 6. Race start logic
    t0 = time.perf_counter()
    if not self.timer.race_started:
        past_threshold = sum(1 for p in positions if p.d_m > 1.0 and p.speed_mps > 2.0 and p.confidence > 0.0)
        self._start_decisions.append(past_threshold)
        if len(self._start_decisions) > 3:
            self._start_decisions.pop(0)
        consistent = sum(1 for v in self._start_decisions if v >= 3) >= 2
        if consistent:
            self.timer.start_race(timestamp)
    current_time = self.timer.get_elapsed(timestamp)
    _N_TIMERS['race_timer'].append(time.perf_counter() - t0)

    # 7. Fill missing lanes
    t0 = time.perf_counter()
    tracked_lanes = {p.lane for p in positions}
    for lane in range(1, 9):
        if lane not in tracked_lanes:
            from tracking.position_estimator import AthletePosition
            positions.append(AthletePosition(
                lane=lane, athlete_id=lane, d_m=0.0, y_world=0.0,
                speed_mps=0.0, timestamp=timestamp,
                frame_count=self.frame_count, confidence=0.0,
            ))
    _N_TIMERS['fill_lanes'].append(time.perf_counter() - t0)

    # 8. Ranking
    t0 = time.perf_counter()
    ranks = self.ranking.compute(positions, current_time)
    _N_TIMERS['ranking'].append(time.perf_counter() - t0)

    # 9. Dynamic camera
    t0 = time.perf_counter()
    if self.dynamic_camera.follow_mode:
        self.dynamic_camera.update(positions)
    _N_TIMERS['dynamic_camera'].append(time.perf_counter() - t0)

    # 10. Occlusion guard + smoother + homography transform
    t0 = time.perf_counter()
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
        if self.frame_tracker is not None:
            u_c, v_c = self.frame_tracker.calib_to_current(smoothed.image.u, smoothed.image.v)
            smoothed.image = ImageCoord(u_c, v_c)
        anchors[lane] = smoothed
    _N_TIMERS['occlusion+smooth'].append(time.perf_counter() - t0)

    # 11. Edge case detection
    t0 = time.perf_counter()
    alerts = self.edge_detector.check_all(athletes, positions)
    _N_TIMERS['edge_detection'].append(time.perf_counter() - t0)

    # 12. Decal rendering (graphic + decal)
    t0 = time.perf_counter()
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
            from rendering.graphic_factory import GraphicContent
            graphic = GraphicContent(rank=rank_entry.rank, time_str=time_str, name=name, lane=lane)
            texture = graphic.render_texture()
            self._texture_cache[lane] = (cache_key, texture)
        from rendering.decal_renderer import DecalInstance
        self.decal_renderer.render_decal(output, DecalInstance(None, anchor, texture))
    _N_TIMERS['decal_rendering'].append(time.perf_counter() - t0)

    # 13. Debug overlay
    t0 = time.perf_counter()
    self.debug_overlay.draw(output, athletes, anchors, self.frame_count, self.fps)
    _N_TIMERS['debug_overlay'].append(time.perf_counter() - t0)

    # 14. Standings
    t0 = time.perf_counter()
    finish_distances = {lane: self.geometry.finish_distance(lane) for lane in range(1, 9)}
    self.standings.draw(output, ranks, positions, self.timer, self.athlete_names, self.geometry.length, timestamp, finish_distances)
    _N_TIMERS['standings'].append(time.perf_counter() - t0)

    # 15. Timer finish
    t0 = time.perf_counter()
    all_finished = len(self.standings.finish_times) >= 8
    if all_finished and not self.timer.race_finished:
        self.timer.finish_race(timestamp)
    _N_TIMERS['timer_finish'].append(time.perf_counter() - t0)

    t_end = time.perf_counter()
    frame_time_ms = (t_end - t_start) * 1000
    self.fps_history.append(1000.0 / max(frame_time_ms, 1))
    if len(self.fps_history) > 30:
        self.fps_history.pop(0)
    self.fps = sum(self.fps_history) / max(len(self.fps_history), 1)
    _N_TIMERS['total_frame'].append(t_end - t_start)
    return output

TrackARPipeline.process_frame = _instrumented_process_frame


def run_profile(geom, r0, t0, cam_K, track_type_label, num_frames=500):
    global _N_TIMERS
    _N_TIMERS.clear()

    pipeline = TrackARPipeline(camera_matrix=cam_K, geometry=geom)

    calib_pts = geom.calibration_world_points()
    w_arr = np.array([w.as_array for w in calib_pts], dtype=np.float64)
    proj, _ = cv2.projectPoints(w_arr, r0, t0, cam_K, np.zeros((4, 1)))
    image_pts = [ImageCoord(float(p[0, 0]), float(p[0, 1])) for p in proj]
    pipeline.calibrate_from_points(calib_pts, image_pts)

    # Dense tracking grid
    track_pts = list(calib_pts)
    for dm in np.arange(0.0, geom.length + 1, 10.0):
        for y in np.arange(0.0, min(10.0, 8 * geom.lane_width), 0.61):
            track_pts.append(WorldCoord(dm, y, 0.0))
    pipeline.projector.set_calibration_world_pts(track_pts)

    render_proj = Projector(cam_K, np.zeros((4, 1)))
    render_proj.set_extrinsics(r0.copy(), t0.copy())
    scene = SyntheticScene(render_proj, geom, speeds=[SPEED] * 8)
    fps = 60.0
    race_len = geom.finish_distance(1)
    max_frames = min(num_frames, int(race_len / SPEED * fps) + 200)

    t_start = time.perf_counter()
    fi = 0
    while fi < max_frames:
        rvec, tvec = r0.copy(), t0.copy()
        render_proj.set_extrinsics(rvec, tvec)
        t = fi / fps
        athletes = scene.update(t)
        canvas = scene.render_background(athletes)
        detections = scene.get_detections(athletes)
        pipeline.process_frame(canvas, timestamp=t, external_detections=detections)
        fi += 1
        if pipeline.timer.race_finished:
            break

    wall_elapsed = time.perf_counter() - t_start
    return fi, wall_elapsed


def print_stats(label, timings, total_frames):
    arr = np.array(timings)
    avg_ms = np.mean(arr) * 1000
    min_ms = np.min(arr) * 1000
    max_ms = np.max(arr) * 1000
    pct = np.mean(arr) / np.mean(_N_TIMERS['total_frame']) * 100 if _N_TIMERS['total_frame'] else 0
    print(f"  {label:<22s}  {avg_ms:8.3f} ms  {min_ms:8.3f} ms  {max_ms:8.3f} ms  {pct:5.1f}%")


if __name__ == '__main__':
    print("=" * 80)
    print("TrackAR Pipeline Profiler")
    print("=" * 80)

    for name_label, geom_type, r0, t0 in [
        ("100m (static camera)", "100m", R0_100M, T0_100M),
    ]:
        print(f"\n--- {name_label} ---")
        geom = TrackGeometry(track_type=geom_type)
        n_frames, wall_time = run_profile(geom, r0, t0, K, geom_type, num_frames=500)
        print(f"  Processed {n_frames} frames in {wall_time:.2f}s wall time")
        avg_total = np.mean(_N_TIMERS['total_frame'])
        print(f"  Average frame time: {avg_total*1000:.3f} ms  ({1/avg_total:.1f} FPS)")

        print()
        print(f"  {'Stage':<22s}  {'Avg':>8s}  {'Min':>8s}  {'Max':>8s}  {'%Total':>6s}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")

        stages = [
            'preprocess', 'camera_tracking', 'detection', 'lane_assignment',
            'position_estimation', 'race_timer', 'fill_lanes', 'ranking',
            'dynamic_camera', 'occlusion+smooth', 'edge_detection',
            'decal_rendering', 'debug_overlay', 'standings', 'timer_finish',
        ]
        for s in stages:
            if s in _N_TIMERS and _N_TIMERS[s]:
                print_stats(s, _N_TIMERS[s], n_frames)

        print()
        print_stats("TOTAL", _N_TIMERS['total_frame'], n_frames)

        # Rank by average time
        averages = []
        for s in stages:
            if s in _N_TIMERS and _N_TIMERS[s]:
                avg = np.mean(_N_TIMERS[s]) * 1000
                averages.append((avg, s))
        averages.sort(reverse=True)

        print()
        print("  Top 3 Bottlenecks:")
        for i, (avg, sname) in enumerate(averages[:3]):
            print(f"    {i+1}. {sname:<22s}  {avg:.3f} ms")

    print("\n" + "=" * 80)
    print("Profile complete.")
    print("=" * 80)
