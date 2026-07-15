import numpy as np

LANE_WIDTH = 1.22
NUM_LANES = 8


class TrackModel:
    INNER_RADIUS = 36.5
    RUNNING_OFFSET = 0.3
    STRAIGHT_LENGTH = 84.39

    def running_radius(self, lane: int) -> float:
        return self.INNER_RADIUS + self.RUNNING_OFFSET + (lane - 1) * LANE_WIDTH

    def curve_arc(self, lane: int) -> float:
        return np.pi * self.running_radius(lane)

    def total_arc_length(self, lane: int) -> float:
        return 2 * self.STRAIGHT_LENGTH + 2 * self.curve_arc(lane)

    def stagger_offset(self, lane: int) -> float:
        return self.total_arc_length(lane) - self.race_distance()

    def race_distance(self) -> float:
        return 400.0

    def _arc_at(self, lane: int, d_m: float) -> float:
        return (d_m + self.stagger_offset(lane)) % self.total_arc_length(lane)

    def get_xy(self, lane: int, d_m: float) -> tuple[float, float]:
        R = self.running_radius(lane)
        S = self.STRAIGHT_LENGTH
        C = self.curve_arc(lane)
        arc = self._arc_at(lane, d_m)

        if arc <= C:
            angle = arc / R
            x = -(S / 2 + R * np.sin(angle))
            y = -(R * np.cos(angle))
        elif arc <= C + S:
            d = arc - C
            x = -S / 2 + d
            y = R
        elif arc <= 2 * C + S:
            d = arc - C - S
            angle = d / R
            x = S / 2 + R * np.sin(angle)
            y = R * np.cos(angle)
        else:
            d = arc - 2 * C - S
            x = S / 2 - d
            y = -R
        return x, y

    def get_heading(self, lane: int, d_m: float) -> float:
        R = self.running_radius(lane)
        S = self.STRAIGHT_LENGTH
        C = self.curve_arc(lane)
        arc = self._arc_at(lane, d_m)

        if arc <= C:
            return np.pi / 2 - arc / R
        elif arc <= C + S:
            return 0.0
        elif arc <= 2 * C + S:
            return -np.pi / 2 + (arc - C - S) / R
        else:
            return np.pi

    def __init__(self):
        self._sample_cache: dict[int, list[tuple[float, float, float]]] = {}

    def _get_samples(self, lane: int) -> list[tuple[float, float, float]]:
        if lane not in self._sample_cache:
            step = 0.3
            rd = self.race_distance()
            n = int(rd / step) + 1
            samples = []
            for i in range(n):
                dm = i * step
                x, y = self.get_xy(lane, dm)
                samples.append((dm, x, y))
            self._sample_cache[lane] = samples
        return self._sample_cache[lane]

    def find_nearest(self, x: float, y: float, max_dist: float = 5.0) -> tuple[int, float, float]:
        best_lane = 1
        best_dm = 0.0
        best_dist = max_dist
        for lane in range(1, NUM_LANES + 1):
            for dm, sx, sy in self._get_samples(lane):
                d2 = (sx - x) * (sx - x) + (sy - y) * (sy - y)
                if d2 < best_dist * best_dist:
                    best_dist = np.sqrt(d2)
                    best_lane = lane
                    best_dm = dm
        return best_lane, np.clip(best_dm, 0.0, self.race_distance()), best_dist
