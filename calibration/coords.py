import numpy as np
from dataclasses import dataclass
from typing import Optional

from calibration.track_model import TrackModel
import calibration.track_model as _tm

TRACK_LENGTH_100M = 100.0
LANE_WIDTH = _tm.LANE_WIDTH
NUM_LANES = _tm.NUM_LANES
TRACK_TOTAL_WIDTH = LANE_WIDTH * NUM_LANES


@dataclass
class WorldCoord:
    x: float
    y: float
    z: float = 0.0

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class ImageCoord:
    u: float
    v: float

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.u, self.v], dtype=np.float64)


@dataclass
class CameraCoord:
    xc: float
    yc: float
    zc: float

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.xc, self.yc, self.zc], dtype=np.float64)


class TrackGeometry:
    def __init__(self, track_type: str = "100m"):
        if track_type not in ("100m", "400m"):
            raise ValueError(f"track_type must be '100m' or '400m', got {track_type!r}")
        self.track_type = track_type
        self.lane_width = LANE_WIDTH
        self.num_lanes = NUM_LANES
        if track_type == "100m":
            self.length = TRACK_LENGTH_100M
            self._model = None
        else:
            self._model = TrackModel()
            self.length = self._model.race_distance()

    def lane_center_y(self, lane: int) -> float:
        if lane < 1 or lane > self.num_lanes:
            raise ValueError(f"Lane must be 1-{self.num_lanes}, got {lane}")
        return (lane - 0.5) * self.lane_width

    def lane_range_y(self, lane: int) -> tuple[float, float]:
        y0 = (lane - 1) * self.lane_width
        y1 = lane * self.lane_width
        return y0, y1

    def lane_from_y(self, y_world: float) -> int:
        lane = int(np.floor(y_world / self.lane_width)) + 1
        lane = np.clip(lane, 1, self.num_lanes)
        return int(lane)

    def world_coord(self, lane: int, d_m: float, lateral_shift: float = 0.0, z: float = 0.0) -> WorldCoord:
        if self._model is not None:
            x, y = self._model.get_xy(lane, d_m)
            return WorldCoord(x, y + lateral_shift, z)
        return WorldCoord(d_m, self.lane_center_y(lane) + lateral_shift, z)

    def find_lane_and_dm(self, x: float, y: float, max_dist: float = 5.0) -> tuple[int, float]:
        if self._model is not None:
            lane, dm, dist = self._model.find_nearest(x, y, max_dist)
            return lane, dm
        lane = self.lane_from_y(y)
        return lane, np.clip(x, 0.0, self.length)

    def distance_to_track(self, x: float, y: float) -> float:
        if self._model is not None:
            _, _, dist = self._model.find_nearest(x, y, max_dist=100.0)
            return dist
        lane = self.lane_from_y(y)
        center_y = self.lane_center_y(lane)
        dy = abs(y - center_y)
        dx = 0.0 if 0 <= x <= self.length else min(abs(x), abs(x - self.length))
        return np.hypot(dx, dy)

    def _project_to_image(self, x: float, y: float, projector) -> tuple[float, float]:
        from calibration.projector import Projector
        wc = WorldCoord(x, y, 0.0)
        ic = projector.project(wc)
        return ic.u, ic.v

    def _build_image_cache(self, projector):
        if self._model is None:
            return
        cache_id = id(projector)
        if getattr(self, '_img_cache_id', None) == cache_id:
            return
        step = 2.0
        rd = self._model.race_distance()
        n = int(rd / step) + 1
        img_samples: dict[int, list[tuple[float, float, float]]] = {}
        for lane in range(1, NUM_LANES + 1):
            samples = []
            for i in range(n):
                dm = i * step
                x, y = self._model.get_xy(lane, dm)
                u, v = self._project_to_image(x, y, projector)
                samples.append((dm, u, v))
            img_samples[lane] = samples
        self._img_cache = img_samples
        self._img_cache_id = cache_id

    def find_lane_dm_from_image(self, u: float, v: float, projector, max_dist_px: float = 150.0) -> tuple[int, float]:
        if self._model is None:
            return 1, np.clip(u, 0.0, self.length)
        self._build_image_cache(projector)
        best_lane = 1
        best_dm = 0.0
        best_dist = max_dist_px
        for lane in range(1, NUM_LANES + 1):
            for dm, su, sv in self._img_cache[lane]:
                d2 = (su - u) * (su - u) + (sv - v) * (sv - v)
                if d2 < best_dist * best_dist:
                    best_dist = np.sqrt(d2)
                    best_lane = lane
                    best_dm = dm
        return best_lane, np.clip(best_dm, 0.0, self.length)

    def race_length(self) -> float:
        return self.length

    def finish_distance(self, lane: int) -> float:
        if self._model is not None:
            return self._model.race_distance()
        return self.length

    def calibration_world_points(self) -> list[WorldCoord]:
        """Returns 4 world coords for PnP calibration:
           [Start×Lane1, Start×Lane8, Finish×Lane1, Finish×Lane8]"""
        start_l1 = self.world_coord(1, 0.0)
        start_l8 = self.world_coord(8, 0.0)
        if self._model is not None:
            m = self._model
            f1 = m.curve_arc(1) + m.STRAIGHT_LENGTH - m.stagger_offset(1)
            f8 = m.curve_arc(8) + m.STRAIGHT_LENGTH - m.stagger_offset(8)
            finish_l1 = self.world_coord(1, f1)
            finish_l8 = self.world_coord(8, f8)
        else:
            finish_l1 = self.world_coord(1, self.length)
            finish_l8 = self.world_coord(8, self.length)
        return [start_l1, start_l8, finish_l1, finish_l8]

    def reference_points_world(self) -> list[tuple[str, WorldCoord]]:
        calib = self.calibration_world_points()
        if self._model is not None:
            return [
                ("start_lane1", calib[0]),
                ("start_lane8", calib[1]),
                ("finish_lane1", calib[2]),
                ("finish_lane8", calib[3]),
                ("mid_lane4", self.world_coord(4, self.length / 2)),
            ]
        return [
            ("start_lane1", calib[0]),
            ("start_lane8", calib[1]),
            ("finish_lane1", calib[2]),
            ("finish_lane8", calib[3]),
            ("mid_lane4", self.world_coord(4, self.length / 2)),
        ]
