import numpy as np
from dataclasses import dataclass
from typing import Optional

from tracking.lane_assigner import AthleteState
from calibration.coords import TrackGeometry


@dataclass
class AthleteAlert:
    lane: int
    alert_type: str
    message: str
    severity: str = "info"


class EdgeCaseDetector:
    def __init__(self, geometry: TrackGeometry):
        self.geometry = geometry
        self.speed_history: dict[int, list[float]] = {}
        self.alert_history: dict[int, list[AthleteAlert]] = {}

    def check_fallen(self, athlete: AthleteState) -> Optional[AthleteAlert]:
        if athlete.detection is None:
            return None
        bbox_h = athlete.detection.height
        if bbox_h < 50:
            return AthleteAlert(
                lane=athlete.lane,
                alert_type="fallen",
                message=f"Lane {athlete.lane}: possible fallen (bbox height {bbox_h:.0f}px)",
                severity="warning",
            )
        return None

    def check_speed_anomaly(self, athlete: AthleteState, speed: float) -> Optional[AthleteAlert]:
        if athlete.lane not in self.speed_history:
            self.speed_history[athlete.lane] = []
        self.speed_history[athlete.lane].append(speed)
        if len(self.speed_history[athlete.lane]) > 30:
            self.speed_history[athlete.lane].pop(0)
        recent = self.speed_history[athlete.lane][-5:]
        if len(recent) >= 3:
            avg_speed = np.mean(recent)
            if speed < avg_speed * 0.3 and avg_speed > 3.0:
                return AthleteAlert(
                    lane=athlete.lane,
                    alert_type="speed_drop",
                    message=f"Lane {athlete.lane}: speed dropped {avg_speed:.1f} -> {speed:.1f} m/s",
                    severity="warning",
                )
        return None

    def check_lane_switch(self, athlete: AthleteState, prev_lane: int) -> Optional[AthleteAlert]:
        if prev_lane != athlete.lane:
            return AthleteAlert(
                lane=athlete.lane,
                alert_type="lane_switch",
                message=f"Athlete {athlete.athlete_id}: crossed from lane {prev_lane} to {athlete.lane}",
                severity="info",
            )
        return None

    def check_finish_line(self, athlete: AthleteState) -> Optional[AthleteAlert]:
        if athlete.d_m >= self.geometry.length - 0.5:
            return AthleteAlert(
                lane=athlete.lane,
                alert_type="finish",
                message=f"Lane {athlete.lane}: finished! ({athlete.d_m:.1f}m)",
                severity="info",
            )
        return None

    def check_all(self, athletes: dict[int, AthleteState],
                  positions: list) -> dict[int, list[AthleteAlert]]:
        alerts: dict[int, list[AthleteAlert]] = {}
        for lane, athlete in athletes.items():
            lane_alerts = []
            fallen = self.check_fallen(athlete)
            if fallen:
                lane_alerts.append(fallen)
            finish = self.check_finish_line(athlete)
            if finish:
                lane_alerts.append(finish)
            if lane_alerts:
                alerts[lane] = lane_alerts
        return alerts
