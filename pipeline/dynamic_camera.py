import numpy as np
from calibration.coords import WorldCoord
from calibration.projector import Projector
from tracking.position_estimator import AthletePosition


class DynamicCamera:
    LOOK_AHEAD_M = 15.0
    FINISH_THRESHOLD_M = 15.0
    SMOOTH_ALPHA = 0.15

    def __init__(self, projector: Projector):
        self.projector = projector
        self.follow_mode = False
        self.prev_look_x: float | None = None
        self.start_x = 0.0
        self.finish_x = 100.0
        self.lane_center_y = 4.88

    def compute_look_x(self, positions: list[AthletePosition]) -> float:
        active = [p for p in positions if p.confidence > 0.1 and p.d_m >= 0.0]
        if not active or not self.follow_mode:
            return self.start_x

        avg_x = float(np.mean([p.d_m for p in active]))
        max_x = float(max(p.d_m for p in active))

        near_finish = any(p.d_m > self.finish_x - self.FINISH_THRESHOLD_M for p in active)
        if near_finish:
            return self.finish_x

        look_x = avg_x + self.LOOK_AHEAD_M
        look_x = min(look_x, self.finish_x - 5.0)
        look_x = max(look_x, self.start_x)
        return look_x

    def update(self, positions: list[AthletePosition]):
        target_x = self.compute_look_x(positions)
        if self.prev_look_x is None:
            self.prev_look_x = target_x
        self.prev_look_x += (target_x - self.prev_look_x) * self.SMOOTH_ALPHA
        self.projector.look_at(WorldCoord(self.prev_look_x, self.lane_center_y, 0.0))
