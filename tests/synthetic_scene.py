import numpy as np
import cv2
from dataclasses import dataclass, field

from calibration.coords import TrackGeometry, WorldCoord
from calibration.projector import Projector
from detection.detector import Detection


@dataclass
class SynthAthleteState:
    lane: int
    speed: float
    d_m: float = 0.0
    finished: bool = False


class SyntheticScene:
    LANE_COLORS = [
        (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
        (255, 255, 0), (255, 165, 0), (255, 0, 0), (128, 0, 128),
    ]

    def __init__(self, projector: Projector, geometry: TrackGeometry, speeds: list[float] | None = None):
        self.projector = projector
        self.geometry = geometry
        self.speeds = speeds or [9.5, 9.8, 10.2, 9.0, 8.5, 9.3, 8.8, 10.5]
        self.h, self.w = 1080, 1920
        self.length = geometry.length
        self._finish_dms: dict[int, float] = {}
        if geometry._model is not None:
            self._race_distance = geometry._model.race_distance()
            for lane in range(1, 9):
                self._finish_dms[lane] = self._race_distance
        self._static_noise: np.ndarray | None = None
        self._track_noise: np.ndarray | None = None

    def set_camera_pose(self, rvec: np.ndarray | None = None, tvec: np.ndarray | None = None):
        if rvec is not None and tvec is not None:
            self.projector.set_extrinsics(rvec, tvec)

    def update(self, t: float):
        athletes = []
        finish_dm = self._race_distance if hasattr(self, '_race_distance') else self.length
        for lane in range(1, 9):
            d_m = max(0, t * self.speeds[lane - 1])
            finished = d_m >= finish_dm
            if self._finish_dms and finished:
                d_m = self._finish_dms[lane]
            athletes.append(SynthAthleteState(lane=lane, speed=self.speeds[lane - 1], d_m=d_m, finished=finished))
        return athletes

    def _get_img_pos(self, athlete: SynthAthleteState):
        return self.projector.project(
            self.geometry.world_coord(athlete.lane, athlete.d_m, z=0.0))

    def render(self, athletes: list[SynthAthleteState]) -> np.ndarray:
        canvas = np.ones((self.h, self.w, 3), dtype=np.uint8) * 50
        # Static noise texture (same every frame) to give ORB stable features
        if self._static_noise is None:
            rng = np.random.RandomState(42)
            noise = rng.randint(-10, 10, (self.h, self.w, 3), dtype=np.int16)
            self._static_noise = noise
        noise = np.clip(self._static_noise.astype(np.int16) + canvas.astype(np.int16), 0, 255).astype(np.uint8)
        canvas = noise
        # Draw faint lane grid lines to give ORB more features
        for x in range(0, self.w, 80):
            cv2.line(canvas, (x, 0), (x, self.h), (48, 48, 48), 1)
        for y in range(0, self.h, 80):
            cv2.line(canvas, (0, y), (self.w, y), (48, 48, 48), 1)
        L = self.length
        is_400m = L > 200

        if is_400m:
            n = 60
            for lane in range(1, 9):
                pts = []
                for i in range(n + 1):
                    dm = L * i / n
                    wc = self.geometry.world_coord(lane, dm)
                    ip = self.projector.project(wc)
                    pts.append([int(ip.u), int(ip.v)])
                pts = np.array(pts, dtype=np.int32).reshape(1, -1, 2)
                shade = 55 if lane % 2 == 0 else 50
                cv2.fillPoly(canvas, pts, (shade, shade, shade))
                if lane < 9:
                    pts_div = []
                    for i in range(n + 1):
                        dm = L * i / n
                        wc = self.geometry.world_coord(lane, dm, lateral_shift=-self.geometry.lane_width/2)
                        ip = self.projector.project(wc)
                        pts_div.append([int(ip.u), int(ip.v)])
                    pts_div = np.array(pts_div, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [pts_div], False, (80, 80, 80), 1)
            pts_finish = []
            m = self.geometry._model
            for lane in range(1, 9):
                fd = m.curve_arc(lane) + m.STRAIGHT_LENGTH - m.stagger_offset(lane)
                wc = self.geometry.world_coord(lane, fd)
                ip = self.projector.project(wc)
                pts_finish.append((int(ip.u), int(ip.v)))
            for i in range(len(pts_finish) - 1):
                cv2.line(canvas, pts_finish[i], pts_finish[i+1], (0, 255, 0), 3)
            pts_start = []
            for lane in range(1, 9):
                wc = self.geometry.world_coord(lane, 0.0)
                ip = self.projector.project(wc)
                pts_start.append((int(ip.u), int(ip.v)))
            for i in range(len(pts_start) - 1):
                cv2.line(canvas, pts_start[i], pts_start[i+1], (0, 0, 255), 3)
            mid = len(pts_start) // 2
            cv2.putText(canvas, "START", (pts_start[mid][0] + 10, pts_start[mid][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(canvas, "FINISH", (pts_finish[mid][0] + 10, pts_finish[mid][1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            for lane in range(1, 9):
                y0 = (lane - 1) * self.geometry.lane_width
                y1 = lane * self.geometry.lane_width
                start_pts = [
                    self.projector.project(WorldCoord(0.0, y0, 0.0)),
                    self.projector.project(WorldCoord(0.0, y1, 0.0)),
                    self.projector.project(WorldCoord(L, y1, 0.0)),
                    self.projector.project(WorldCoord(L, y0, 0.0)),
                ]
                pts = np.array([[int(p.u), int(p.v)] for p in start_pts], dtype=np.int32).reshape(1, -1, 2)
                shade = 60 if lane % 2 == 0 else 50
                cv2.fillPoly(canvas, pts, (shade, shade, shade))
            for boundary_idx in range(9):
                y_w = boundary_idx * self.geometry.lane_width
                p0 = self.projector.project(WorldCoord(0.0, y_w, 0.0))
                p1 = self.projector.project(WorldCoord(L, y_w, 0.0))
                cv2.line(canvas, (int(p0.u), int(p0.v)), (int(p1.u), int(p1.v)), (80, 80, 80), 2)
            s0 = self.projector.project(WorldCoord(0.0, 0.0, 0.0))
            s1 = self.projector.project(WorldCoord(0.0, self.geometry.lane_width * 8, 0.0))
            cv2.line(canvas, (int(s0.u), int(s0.v)), (int(s1.u), int(s1.v)), (0, 0, 255), 4)
            cv2.putText(canvas, "START", (int(s0.u) - 140, int((s0.v + s1.v) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            f0 = self.projector.project(WorldCoord(L, 0.0, 0.0))
            f1 = self.projector.project(WorldCoord(L, self.geometry.lane_width * 8, 0.0))
            cv2.line(canvas, (int(f0.u), int(f0.v)), (int(f1.u), int(f1.v)), (0, 255, 0), 4)
            cv2.putText(canvas, "FINISH", (int(f1.u) + 15, int((f0.v + f1.v) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            for lane in range(1, 9):
                p = self.projector.project(WorldCoord(-2.0, self.geometry.lane_center_y(lane), 0.0))
                cv2.putText(canvas, f"L{lane}", (int(p.u) - 15, int(p.v) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        for a in athletes:
            ip = self._get_img_pos(a)
            cx, cy = int(ip.u), int(ip.v)
            color = self.LANE_COLORS[a.lane - 1]
            cv2.circle(canvas, (cx, cy), 14, color, -1)
            cv2.rectangle(canvas, (cx - 11, cy - 24), (cx + 11, cy + 24), color, 2)
        return canvas

    def render_background(self, athletes: list[SynthAthleteState]) -> np.ndarray:
        """Render track background with dense features for KLT/ORB tracking."""
        rng = np.random.RandomState(42)
        canvas = np.ones((self.h, self.w, 3), dtype=np.uint8) * 55
        L = self.length
        is_400m = L > 200

        # --- Lane polygons with subtle per-lane shading + noise texture ---
        if is_400m:
            n = 60
            # Pre-compute all world coords for lane fill + dividers
            lane_fill_pts: list[list[WorldCoord]] = []
            lane_div_pts: list[list[WorldCoord]] = []
            for lane in range(1, 9):
                fill = [WorldCoord(L * i / n, self.geometry.world_coord(lane, L * i / n).y, 0.0) for i in range(n + 1)]
                lane_fill_pts.append(fill)
                if lane < 9:
                    div = [WorldCoord(L * i / n, self.geometry.world_coord(lane, L * i / n, lateral_shift=-self.geometry.lane_width/2).y, 0.0) for i in range(n + 1)]
                    lane_div_pts.append(div)
            # Batch project all fill points
            all_fill = [p for seg in lane_fill_pts for p in seg]
            all_fill_img = self.projector.project_batch(all_fill)
            idx = 0
            for lane_idx, seg in enumerate(lane_fill_pts):
                lane = lane_idx + 1
                pts_np = np.array([[int(all_fill_img[idx + i].u), int(all_fill_img[idx + i].v)] for i in range(len(seg))], dtype=np.int32).reshape(1, -1, 2)
                base = 62 if lane % 2 == 0 else 50
                shade = max(0, min(255, base + int(rng.randint(-3, 4))))
                cv2.fillPoly(canvas, pts_np, (shade, shade, shade))
                idx += len(seg)
            # Batch project divider points
            all_div = [p for seg in lane_div_pts for p in seg]
            all_div_img = self.projector.project_batch(all_div)
            idx = 0
            for seg in lane_div_pts:
                pts_np = np.array([[int(all_div_img[idx + i].u), int(all_div_img[idx + i].v)] for i in range(len(seg))], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts_np], False, (80, 80, 80), 2)
                idx += len(seg)
        else:
            # Batch project lane polygon corners
            lane_polys = []
            for lane in range(1, 9):
                y0 = (lane - 1) * self.geometry.lane_width
                y1 = lane * self.geometry.lane_width
                lane_polys.append([WorldCoord(0.0, y0, 0.0), WorldCoord(0.0, y1, 0.0),
                                   WorldCoord(L, y1, 0.0), WorldCoord(L, y0, 0.0)])
            all_poly_pts = [p for seg in lane_polys for p in seg]
            poly_img = self.projector.project_batch(all_poly_pts)
            idx = 0
            for lane in range(1, 9):
                pts_np = np.array([[int(poly_img[idx].u), int(poly_img[idx].v)],
                                   [int(poly_img[idx+1].u), int(poly_img[idx+1].v)],
                                   [int(poly_img[idx+2].u), int(poly_img[idx+2].v)],
                                   [int(poly_img[idx+3].u), int(poly_img[idx+3].v)]], dtype=np.int32).reshape(1, -1, 2)
                base = 64 if lane % 2 == 0 else 52
                shade = max(0, min(255, base + int(rng.randint(-4, 5))))
                cv2.fillPoly(canvas, pts_np, (shade, shade, shade))
                idx += 4
            # Batch project lane divider lines
            div_pts = [WorldCoord(0.0, bi * self.geometry.lane_width, 0.0) for bi in range(9)]
            div_pts.extend([WorldCoord(L, bi * self.geometry.lane_width, 0.0) for bi in range(9)])
            div_img = self.projector.project_batch(div_pts)
            for bi in range(9):
                p0 = div_img[bi]
                p1 = div_img[9 + bi]
                cv2.line(canvas, (int(p0.u), int(p0.v)), (int(p1.u), int(p1.v)), (140, 140, 140), 3)
            # Start line (red)
            s_pts = self.projector.project_batch([WorldCoord(0.0, 0.0, 0.0), WorldCoord(0.0, self.geometry.lane_width * 8, 0.0)])
            cv2.line(canvas, (int(s_pts[0].u), int(s_pts[0].v)), (int(s_pts[1].u), int(s_pts[1].v)), (0, 0, 200), 4)
            # Finish line (green)
            f_pts = self.projector.project_batch([WorldCoord(L, 0.0, 0.0), WorldCoord(L, self.geometry.lane_width * 8, 0.0)])
            cv2.line(canvas, (int(f_pts[0].u), int(f_pts[0].v)), (int(f_pts[1].u), int(f_pts[1].v)), (0, 180, 0), 4)

        # --- Track surface noise texture (realistic) ---
        if self._track_noise is None:
            rng = np.random.RandomState(42)
            # Perlin-like multi-scale noise
            c1 = rng.randn(self.h // 16, self.w // 16).astype(np.float32)
            c1 = cv2.resize(c1, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            c1 = cv2.GaussianBlur(c1, (31, 31), 8.0)
            c2 = rng.randn(self.h // 8, self.w // 8).astype(np.float32)
            c2 = cv2.resize(c2, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            c2 = cv2.GaussianBlur(c2, (15, 15), 4.0)
            c3 = rng.randn(self.h // 4, self.w // 4).astype(np.float32)
            c3 = cv2.resize(c3, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            c3 = cv2.GaussianBlur(c3, (7, 7), 2.0)
            noise_mono = (c1 * 5 + c2 * 3 + c3 * 2).astype(np.int16)
            self._track_noise = np.stack([noise_mono] * 3, axis=-1)

        canvas = np.clip(canvas.astype(np.int16) + self._track_noise, 20, 140).astype(np.uint8)
        return canvas

    def get_detections(self, athletes: list[SynthAthleteState]) -> list[Detection]:
        detections = []
        for a in athletes:
            ip = self._get_img_pos(a)
            h_px = 80
            w_px = 40
            cx = int(ip.u)
            cy = int(ip.v)
            x1 = cx - w_px // 2
            y1 = cy - h_px
            x2 = cx + w_px // 2
            y2 = cy
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.w, x2)
            y2 = min(self.h, y2)
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=0.95))
        return detections
