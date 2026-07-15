import numpy as np
from dataclasses import dataclass

from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from tracking.lane_assigner import AthleteState


@dataclass
class AthletePosition:
    lane: int
    athlete_id: int
    d_m: float
    y_world: float
    speed_mps: float
    timestamp: float
    frame_count: int
    confidence: float = 0.0


class PositionEstimator:
    MAX_SPEED_MPS = 15.0
    MAX_JUMP_M = 5.0

    def __init__(self, geometry: TrackGeometry, projector: Projector):
        self.geometry = geometry
        self.projector = projector
        self.frame_count = 0
        self.prev_positions: dict[int, AthletePosition] = {}

    def estimate(self, athletes: dict[int, AthleteState], timestamp: float) -> list[AthletePosition]:
        self.frame_count += 1
        positions = []
        for lane, state in athletes.items():
            raw_dm = state.d_m
            speed = 0.0
            confidence = 1.0
            prev = self.prev_positions.get(lane)

            if prev is not None:
                dt = timestamp - prev.timestamp
                if dt > 0:
                    speed = (raw_dm - prev.d_m) / dt

                if state.frames_missed > 0:
                    confidence = max(0.0, 1.0 - state.frames_missed * 0.2)
                else:
                    jump = abs(raw_dm - prev.d_m)
                    if jump > self.MAX_JUMP_M and prev.d_m > 1.0:
                        expected_speed = prev.speed_mps
                        expected_jump = expected_speed * dt
                        max_allowed = max(expected_jump + 2.0, self.MAX_JUMP_M)
                        if jump > max_allowed:
                            raw_dm = prev.d_m + np.sign(raw_dm - prev.d_m) * max_allowed
                            speed = expected_speed
                            confidence = 0.3

            speed = np.clip(speed, -self.MAX_SPEED_MPS, self.MAX_SPEED_MPS)
            raw_dm = np.clip(raw_dm, 0.0, self.geometry.length)

            pos = AthletePosition(
                lane=lane,
                athlete_id=state.athlete_id,
                d_m=raw_dm,
                y_world=state.y_world,
                speed_mps=speed,
                timestamp=timestamp,
                frame_count=self.frame_count,
                confidence=confidence,
            )
            positions.append(pos)
            self.prev_positions[lane] = pos
        return positions
