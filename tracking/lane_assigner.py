import cv2
import numpy as np
from dataclasses import dataclass, field

from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from detection.detector import Detection
from tracking.kalman import LaneKalmanFilter


@dataclass
class AthleteState:
    lane: int
    athlete_id: int
    d_m: float = 0.0
    y_world: float = 0.0
    detection: Detection | None = None
    kalman: LaneKalmanFilter = field(default_factory=LaneKalmanFilter)
    frames_missed: int = 0
    frames_tracked: int = 0
    dm_min: float = 0.0
    dm_max: float = 0.0
    is_active: bool = True
    speed_mps: float = 0.0
    tracking_confidence: float = 0.0
    # Smooth pixel position for matching
    last_px: tuple[float, float] = (0.0, 0.0)
    # Per-lane occlusion hint: True if another athlete may occlude this one
    occluded: bool = False
    # How many consecutive frames this athlete has been coasting
    coast_count: int = 0
    # Expected bbox size (for false positive rejection)
    expected_bbox_area: float = 0.0
    # Recent bbox width/height history
    bbox_history: list[tuple[float, float]] = field(default_factory=list)
    # Bias-free distance traveled (integrated from unbiased speed)
    distance_traveled: float = 0.0


class LaneAssigner:
    MIN_CONFIDENCE = 0.25
    MIN_CONFIDENCE_NEW = 0.5
    MAX_DM_JUMP = 10.0
    MAX_SPEED_MPS = 15.0
    MAX_DIST_PX = 100.0
    OFF_TRACK_DIST_PX = 200.0
    DUPLICATE_PX_DIST = 40.0
    MAX_MATCH_SCORE = 10.0

    def __init__(self, geometry: TrackGeometry, projector: Projector,
                 max_missed_frames: int = 600):
        self.geometry = geometry
        self.projector = projector
        self.max_missed_frames = max_missed_frames
        self.athletes: dict[int, AthleteState] = {}
        self.next_id = 1
        self._img_cache_id = None
        self._img_samples: dict[int, list[tuple[float, float, float]]] = {}
        self._inner_hull: np.ndarray | None = None
        self._outer_hull: np.ndarray | None = None
        self._H_calib_curr: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None
        self._pending: dict[int, dict] = {}
        self._blacklist: dict[int, int] = {}
        self._frame_count = 0
        # Track region for 100m: computed from calibration
        self._track_bbox_100m: tuple[float, float, float, float] | None = None
        # World-to-image homography for 100m dm estimation
        self._world_to_image_H: np.ndarray | None = None

    def set_H_calib_current(self, H: np.ndarray | None):
        self._H_calib_curr = H
        if H is not None:
            try:
                self._H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                self._H_inv = np.eye(3, dtype=np.float64)
        else:
            self._H_inv = None

    def set_track_bbox_100m(self, bbox: tuple[float, float, float, float] | None):
        self._track_bbox_100m = bbox

    def set_world_to_image_H(self, H: np.ndarray | None):
        self._world_to_image_H = H

    def _current_to_calib(self, u: float, v: float) -> tuple[float, float]:
        if self._H_inv is None:
            return u, v
        pt = np.array([[[u, v]]], dtype=np.float64)
        tr = cv2.perspectiveTransform(pt, self._H_inv)
        return float(tr[0, 0, 0]), float(tr[0, 0, 1])

    def _calib_to_current(self, u: float, v: float) -> tuple[float, float]:
        if self._H_calib_curr is None:
            return u, v
        pt = np.array([[[u, v]]], dtype=np.float64)
        tr = cv2.perspectiveTransform(pt, self._H_calib_curr)
        return float(tr[0, 0, 0]), float(tr[0, 0, 1])

    def reset(self):
        self.athletes.clear()
        self.next_id = 1
        self._inner_hull = None
        self._outer_hull = None
        self._pending.clear()
        self._blacklist.clear()
        self._frame_count = 0
        self._track_bbox_100m = None

    def preinitialize_athletes(self, lanes: list[int], start_positions: dict[int, float] | None = None):
        """Pre-populate athletes for all lanes so pending-track is not required."""
        is_400m = self.geometry.length > 200
        for lane in lanes:
            world_y = self.geometry.lane_center_y(lane) if is_400m else 0.0
            new = AthleteState(
                lane=lane,
                athlete_id=self.next_id,
                d_m=0.0,
                dm_min=0.0,
                dm_max=0.0,
                y_world=world_y,
                detection=None,
                tracking_confidence=0.5,
            )
            new.kalman.initialize(np.array([0.0]))
            self.athletes[lane] = new
            self.next_id += 1

    def _build_image_cache(self):
        if self.geometry._model is None:
            return
        if getattr(self, '_cache_frame', -1) == self._frame_count:
            return
        self._cache_frame = self._frame_count
        step = 2.0
        rd = self.geometry._model.race_distance()
        n = int(rd / step) + 1
        self._img_dms: dict[int, np.ndarray] = {}
        all_wcs = []
        lane_ranges = []
        for lane in range(1, 9):
            dm_list = np.arange(n, dtype=np.float64) * step
            self._img_dms[lane] = dm_list
            wcs = [self.geometry.world_coord(lane, float(dm_list[i])) for i in range(n)]
            all_wcs.extend(wcs)
            lane_ranges.append((len(all_wcs) - n, len(all_wcs)))
        all_img = self.projector.project_batch(all_wcs)
        self._img_pts: dict[int, np.ndarray] = {}
        for lane, (start, end) in enumerate(lane_ranges, start=1):
            pts = np.array([[all_img[i].u, all_img[i].v] for i in range(start, end)], dtype=np.float64)
            self._img_pts[lane] = pts
        all_pts_cat = np.concatenate(list(self._img_pts.values()), axis=0).astype(np.int32).reshape(-1, 1, 2)
        lane1_pts = self._img_pts[1].astype(np.int32).reshape(-1, 1, 2)
        self._outer_hull = cv2.convexHull(all_pts_cat)
        self._inner_hull = cv2.convexHull(lane1_pts)

    MARGIN_PX = 20.0

    def _is_in_track_region(self, u: float, v: float) -> bool:
        if self._outer_hull is None and self._track_bbox_100m is None:
            return True
        if self._outer_hull is not None:
            pt = (float(u), float(v))
            outer_dist = cv2.pointPolygonTest(self._outer_hull, pt, True)
            if outer_dist < -self.MARGIN_PX:
                return False
            if self._inner_hull is not None:
                inner_dist = cv2.pointPolygonTest(self._inner_hull, pt, True)
                if inner_dist > self.MARGIN_PX:
                    return False
            return True
        if self._track_bbox_100m is not None:
            x0, y0, x1, y1 = self._track_bbox_100m
            margin = self.MARGIN_PX
            return (x0 - margin <= u <= x1 + margin and
                    y0 - margin <= v <= y1 + margin)
        return True

    # ── Audience / Off-Track filter ─────────────────────────────────────
    MIN_ATHLETE_HEIGHT = 40
    MAX_ATHLETE_ASPECT = 2.5
    MIN_ATHLETE_ASPECT = 0.3

    def _is_likely_athlete(self, det: Detection) -> bool:
        h = det.height
        w = det.width
        if h < self.MIN_ATHLETE_HEIGHT:
            return False
        aspect = h / max(w, 1.0)
        if aspect > self.MAX_ATHLETE_ASPECT or aspect < self.MIN_ATHLETE_ASPECT:
            return False
        return True

    # ── Matching helpers ────────────────────────────────────────────────

    def _find_lane_dm_from_image(self, u: float, v: float) -> tuple[int, float, float]:
        if self.geometry._model is None:
            world = self.projector.unproject_to_ground(ImageCoord(u=u, v=v))
            lane = self.geometry.lane_from_y(world.y)
            return lane, np.clip(world.x, 0.0, self.geometry.length), abs(world.y - self.geometry.lane_center_y(lane))
        best_lane = 1
        best_dm = 0.0
        best_dist = 1e9
        for lane in range(1, 9):
            pts = self._img_pts[lane]
            d2 = (pts[:, 0] - u) ** 2 + (pts[:, 1] - v) ** 2
            min_idx = np.argmin(d2)
            min_d = np.sqrt(d2[min_idx])
            if min_d < best_dist:
                best_dist = min_d
                best_lane = lane
                best_dm = float(self._img_dms[lane][min_idx])
        return best_lane, np.clip(best_dm, 0.0, self.geometry.length), best_dist

    def _find_dm_on_lane(self, u: float, v: float, lane: int) -> tuple[float, float]:
        if self.geometry._model is None:
            world = self.projector.unproject_to_ground(ImageCoord(u=u, v=v))
            return np.clip(world.x, 0.0, self.geometry.length), abs(world.y - self.geometry.lane_center_y(lane))
        pts = self._img_pts[lane]
        d2 = (pts[:, 0] - u) ** 2 + (pts[:, 1] - v) ** 2
        min_idx = np.argmin(d2)
        best_dm = float(self._img_dms[lane][min_idx])
        best_dist = np.sqrt(d2[min_idx])
        return np.clip(best_dm, 0.0, self.geometry.length), best_dist

    def _get_predicted_dm(self, athlete: AthleteState) -> float:
        pos = athlete.kalman.get_position()
        dm = pos[0]
        race_len = self.geometry.length
        prev = athlete.d_m
        if dm < 5.0 and prev > race_len - 10.0:
            dm = race_len
        elif dm > race_len - 5.0 and prev < 10.0:
            dm = 0.0
        return np.clip(dm, 0.0, race_len)

    def _predict_pixel_current(self, athlete: AthleteState) -> tuple[float, float]:
        dm = self._get_predicted_dm(athlete)
        wc = self.geometry.world_coord(athlete.lane, dm)
        ic = self.projector.project(wc)
        u, v = ic.u, ic.v
        if self._H_calib_curr is not None:
            pt = np.array([[[u, v]]], dtype=np.float64)
            tr = cv2.perspectiveTransform(pt, self._H_calib_curr)
            u, v = float(tr[0, 0, 0]), float(tr[0, 0, 1])
        return u, v

    MATCH_MAX_PX_DIST = 200.0

    def _match_existing_athlete(self, athlete: AthleteState, detections: list[Detection],
                                 frame_dt: float,
                                 dets_used: set[int] | None = None) -> tuple[Detection, float] | None:
        if dets_used is None:
            dets_used = set()
        is_400m = self.geometry.length > 200
        pu, pv = self._predict_pixel_current(athlete)
        prev_dm = athlete.d_m
        prev_speed = athlete.speed_mps
        prev_conf = athlete.tracking_confidence
        expected_jump = prev_speed * frame_dt
        dt_factor = max(frame_dt * 60.0, 1.0)

        # Progressive search radius: widen if athlete has been coasting
        coast_factor = 1.0 + athlete.coast_count * 0.3
        max_px = self.MATCH_MAX_PX_DIST * coast_factor
        max_jump_base = max(expected_jump + 2.0 + (1.0 - prev_conf) * 5.0,
                            self.MAX_DM_JUMP * dt_factor)
        # Wider dm tolerance for coasting athletes
        max_jump = max_jump_base * coast_factor

        best_det = None
        best_dm = 0.0
        best_score = float('inf')
        for di, det in enumerate(detections):
            if di in dets_used:
                continue
            du, dv = det.bottom_center
            px_d2 = (du - pu) * (du - pu) + (dv - pv) * (dv - pv)
            if px_d2 > max_px * max_px:
                continue
            cu, cv = self._current_to_calib(du, dv)
            dm, _ = self._find_dm_on_lane(cu, cv, athlete.lane)
            if dm < 5.0 and prev_dm > self.geometry.length - 15.0:
                dm = self.geometry.length
            dm_jump = abs(dm - prev_dm)
            if dm_jump > max_jump:
                continue
            # Score: weighted combo of pixel dist + dm jump + confidence + bbox consistency
            score = px_d2 * 0.5 + dm_jump * 3.0 + (1.0 - det.confidence) * 5.0
            if athlete.frames_tracked > 10 and athlete.bbox_history:
                avg_h = np.mean([b[1] for b in athlete.bbox_history[-10:]])
                if avg_h > 0:
                    h_ratio = det.height / avg_h
                    if h_ratio < 0.4 or h_ratio > 2.5:
                        score += 80.0
                    elif h_ratio < 0.6 or h_ratio > 1.8:
                        score += 30.0
            if athlete.tracking_confidence > 0.7 and det.confidence < 0.5:
                score += 40.0
            if score < best_score:
                best_score = score
                best_det = det
                best_dm = dm
        if best_det is not None:
            use_strict = len(detections) > 15 and athlete.frames_tracked > 0
            if use_strict:
                score_thresh = self.MAX_MATCH_SCORE * (1.0 + (1.0 - athlete.tracking_confidence) * 2.0)
                if athlete.frames_tracked < 15:
                    score_thresh *= 2.0
            else:
                score_thresh = 1e9
            if best_score < score_thresh:
                return best_det, best_dm
        return None

    # ── Main frame processing ───────────────────────────────────────────

    def process_frame(self, detections: list[Detection],
                      frame_dt: float = 1.0 / 60.0) -> dict[int, AthleteState]:
        is_400m = self.geometry.length > 200
        if is_400m:
            self._build_image_cache()

        # Filter detections: remove off-track and non-athlete shapes
        filtered_dets = [d for d in detections if self._is_likely_athlete(d)]
        if not is_400m:
            filtered_dets = [
                d for d in filtered_dets
                if self._is_in_track_region(*self._current_to_calib(*d.bottom_center))
            ]
        # NMS: suppress overlapping detections (same-athlete duplicates)
        if len(filtered_dets) > 1:
            filtered_dets.sort(key=lambda d: d.confidence, reverse=True)
            keep = [True] * len(filtered_dets)
            for i in range(len(filtered_dets)):
                if not keep[i]:
                    continue
                bi = filtered_dets[i].bbox
                ai_x1, ai_y1, ai_x2, ai_y2 = bi
                ai_area = max((ai_x2 - ai_x1), 1) * max((ai_y2 - ai_y1), 1)
                for j in range(i + 1, len(filtered_dets)):
                    if not keep[j]:
                        continue
                    bj = filtered_dets[j].bbox
                    x1 = max(ai_x1, bj[0])
                    y1 = max(ai_y1, bj[1])
                    x2 = min(ai_x2, bj[2])
                    y2 = min(ai_y2, bj[3])
                    if x2 < x1 or y2 < y1:
                        continue
                    inter = (x2 - x1) * (y2 - y1)
                    aj_area = max((bj[2] - bj[0]), 1) * max((bj[3] - bj[1]), 1)
                    iou = inter / (ai_area + aj_area - inter + 1e-6)
                    if iou > 0.85:
                        keep[j] = False
            filtered_dets = [d for d, k in zip(filtered_dets, keep) if k]

        for athlete in self.athletes.values():
            athlete.kalman.set_dt(frame_dt)
            athlete.kalman.predict()
            athlete.frames_missed += 1

        matched_lanes: set[int] = set()

        # 1. Match existing athletes via prediction-guided pixel search
        dets_used: set[int] = set()
        match_order = sorted(self.athletes.keys(),
                             key=lambda l: (self.athletes[l].coast_count,
                                            -self.athletes[l].tracking_confidence),
                             reverse=True)
        for lane in match_order:
            athlete = self.athletes[lane]
            result = self._match_existing_athlete(athlete, filtered_dets, frame_dt, dets_used)
            if result is not None:
                det, dm = result
                det_idx = next(i for i, d in enumerate(filtered_dets) if d is det)
                dets_used.add(det_idx)
                prev_dm = athlete.d_m
                athlete.d_m = dm
                athlete.dm_min = min(athlete.dm_min, dm)
                athlete.dm_max = max(athlete.dm_max, dm)
                athlete.detection = det
                # Use innovation-gated Kalman update with confidence
                athlete.kalman.update(np.array([dm]), confidence=athlete.tracking_confidence)
                athlete.kalman.x[0, 0] = dm
                new_speed = abs(dm - prev_dm) / max(frame_dt, 0.001)
                athlete.speed_mps = np.clip(
                    athlete.speed_mps * 0.7 + new_speed * 0.3,
                    -self.MAX_SPEED_MPS, self.MAX_SPEED_MPS)
                athlete.distance_traveled += abs(dm - prev_dm)
                athlete.frames_missed = 0
                athlete.frames_tracked += 1
                athlete.tracking_confidence = min(1.0, athlete.tracking_confidence + 0.1)
                du, dv = det.bottom_center
                athlete.last_px = (du, dv)
                athlete.bbox_history.append((det.width, det.height))
                if len(athlete.bbox_history) > 30:
                    athlete.bbox_history.pop(0)
                matched_lanes.add(lane)

        # 1.5 Fallback: re-acquire unmatched athletes with wider search
        for lane in list(self.athletes.keys()):
            if lane in matched_lanes:
                continue
            athlete = self.athletes[lane]
            # Progressive recovery: wider search for longer-missing athletes
            coast_factor = 1.0 + athlete.coast_count * 0.5
            best = None
            best_score = float('inf')
            for i, det in enumerate(filtered_dets):
                if i in dets_used:
                    continue
                u, v = det.bottom_center
                cu, cv = self._current_to_calib(u, v)
                if not self._is_in_track_region(cu, cv):
                    continue
                _, dm, _ = self._find_lane_dm_from_image(cu, cv)
                if is_400m and dm < 5.0 and athlete.d_m > self.geometry.length - 15.0:
                    dm = self.geometry.length
                dm_jump = abs(dm - athlete.d_m)
                max_allowed = self.MAX_DM_JUMP * 1.5 * coast_factor
                if dm_jump > max_allowed and not is_400m:
                    continue
                score = dm_jump * 3.0 + (1.0 - det.confidence) * 5.0
                if score < best_score:
                    best_score = score
                    best = (det, dm)
            max_score = 500.0 * coast_factor if is_400m else 20.0 * coast_factor
            if best is not None and best_score < max_score:
                det, dm = best
                det_idx = next(i for i, d in enumerate(filtered_dets) if d is det)
                dets_used.add(det_idx)
                prev_dm = athlete.d_m
                athlete.d_m = dm
                athlete.dm_min = min(athlete.dm_min, dm)
                athlete.dm_max = max(athlete.dm_max, dm)
                athlete.detection = det
                athlete.kalman.update(np.array([dm]), confidence=athlete.tracking_confidence)
                athlete.kalman.x[0, 0] = dm
                new_speed = abs(dm - prev_dm) / max(frame_dt, 0.001)
                athlete.speed_mps = np.clip(
                    athlete.speed_mps * 0.7 + new_speed * 0.3,
                    -self.MAX_SPEED_MPS, self.MAX_SPEED_MPS)
                athlete.distance_traveled += abs(dm - prev_dm)
                athlete.frames_missed = 0
                athlete.coast_count = 0
                athlete.frames_tracked += 1
                athlete.tracking_confidence = max(athlete.tracking_confidence, 0.3)
                du, dv = det.bottom_center
                athlete.last_px = (du, dv)
                matched_lanes.add(lane)

        # Update coast_count based on actual consecutive misses
        for a in self.athletes.values():
            if a.frames_missed > 0:
                a.coast_count += 1
            else:
                a.coast_count = 0

        # 2. Filter unused detections for new athletes (with duplicate guard)
        unused_dets = []
        for i, d in enumerate(filtered_dets):
            if i in dets_used:
                continue
            conf_ok = d.confidence >= (self.MIN_CONFIDENCE_NEW if is_400m else self.MIN_CONFIDENCE)
            if not conf_ok:
                continue
            is_dup = False
            u_raw, v_raw = d.bottom_center
            for athlete in self.athletes.values():
                pu, pv = self._predict_pixel_current(athlete)
                dx = u_raw - pu
                dy = v_raw - pv
                if dx * dx + dy * dy < self.DUPLICATE_PX_DIST * self.DUPLICATE_PX_DIST:
                    is_dup = True
                    break
            if not is_dup:
                unused_dets.append(d)

        lane_new_dets: dict[int, list[tuple[Detection, float]]] = {}
        for det in unused_dets:
            u_raw, v_raw = det.bottom_center
            u_foot, v_foot = self._current_to_calib(u_raw, v_raw)
            if not self._is_in_track_region(u_foot, v_foot):
                continue
            if is_400m:
                lane, nearest_dm, pixel_dist = self._find_lane_dm_from_image(u_foot, v_foot)
                if pixel_dist > self.MAX_DIST_PX:
                    continue
                if lane not in self.athletes and nearest_dm > self.geometry.length - 10.0:
                    nearest_dm = 0.0
                if lane in self.athletes:
                    continue
            else:
                world = self.projector.unproject_to_ground(ImageCoord(u=u_foot, v=v_foot))
                lane = self.geometry.lane_from_y(world.y)
                nearest_dm = np.clip(world.x, 0.0, self.geometry.length)
                if lane in self.athletes:
                    continue
            if lane not in lane_new_dets:
                lane_new_dets[lane] = []
            lane_new_dets[lane].append((det, nearest_dm))

        # 3. Manage pending tracks
        for lane in list(self._pending.keys()):
            if lane in self.athletes:
                del self._pending[lane]

        for lane, det_dm_list in lane_new_dets.items():
            if lane in matched_lanes:
                continue
            if lane in self._pending:
                det, dm = det_dm_list[0]
                prev_dm = self._pending[lane]["dm"]
                if abs(dm - prev_dm) < 20.0:
                    self._pending[lane]["count"] += 1
                    self._pending[lane]["dm"] = dm
                else:
                    self._pending[lane]["count"] = 1
                    self._pending[lane]["dm"] = dm
                self._pending[lane]["missed"] = 0
            else:
                det, dm = det_dm_list[0]
                self._pending[lane] = {"count": 1, "dm": dm, "dm_start": dm, "missed": 0}

        for lane in list(self._pending.keys()):
            if lane not in lane_new_dets and lane not in self.athletes:
                self._pending[lane]["missed"] += 1
                if self._pending[lane]["missed"] > 3:
                    del self._pending[lane]

        for lane in list(self._pending.keys()):
            if lane in self.athletes:
                del self._pending[lane]
                continue
            info = self._pending[lane]
            if info["count"] >= 2:
                world_y = self.geometry.lane_center_y(lane) if is_400m else 0.0
                dm = info["dm"]
                new = AthleteState(
                    lane=lane,
                    athlete_id=self.next_id,
                    d_m=dm,
                    dm_min=dm,
                    dm_max=dm,
                    y_world=world_y,
                    detection=None,
                    tracking_confidence=0.5,
                )
                new.kalman.initialize(np.array([dm]))
                self.athletes[lane] = new
                self.next_id += 1
                matched_lanes.add(lane)
                del self._pending[lane]

        self._frame_count += 1

        # 4. Coast unmatched athletes with prediction
        for athlete in self.athletes.values():
            if athlete.lane not in matched_lanes:
                predicted = self._get_predicted_dm(athlete)
                athlete.d_m = predicted
                athlete.kalman.x[0, 0] = float(np.clip(athlete.kalman.x[0, 0], 0.0, self.geometry.length))
                vel = athlete.kalman.get_velocity()
                athlete.distance_traveled += abs(vel) * frame_dt
                # Faster confidence decay during extended coast
                decay = 0.05 if athlete.coast_count < 10 else 0.1
                athlete.tracking_confidence = max(0.0, athlete.tracking_confidence - decay)

        # 4.5 Remove athletes that have been coasting too long
        for lane in list(self.athletes.keys()):
            a = self.athletes[lane]
            if a.frames_missed > self.max_missed_frames:
                a.is_active = False
                del self.athletes[lane]

        for lane in list(self._blacklist.keys()):
            if self._frame_count >= self._blacklist[lane]:
                del self._blacklist[lane]

        for lane in list(self._pending.keys()):
            if lane in self._blacklist:
                del self._pending[lane]

        return self.athletes
