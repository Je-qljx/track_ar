import numpy as np
from dataclasses import dataclass

from calibration.coords import WorldCoord


@dataclass
class DepthLayer:
    world_z: float
    layer_name: str
    order: int


class DepthSorter:
    LAYERS = {
        "track_ground": DepthLayer(world_z=1000.0, layer_name="track_ground", order=0),
        "far_lane_graphic": DepthLayer(world_z=900.0, layer_name="far_lane_graphic", order=1),
        "far_lane_athlete": DepthLayer(world_z=800.0, layer_name="far_lane_athlete", order=2),
        "near_lane_graphic": DepthLayer(world_z=200.0, layer_name="near_lane_graphic", order=3),
        "near_lane_athlete": DepthLayer(world_z=100.0, layer_name="near_lane_athlete", order=4),
    }

    @staticmethod
    def compute_depth(world_pos: WorldCoord, camera_origin: WorldCoord = WorldCoord(0, -15, 20)) -> float:
        dx = world_pos.x - camera_origin.x
        dy = world_pos.y - camera_origin.y
        dz = (world_pos.z or 0) - camera_origin.z
        return float(np.sqrt(dx**2 + dy**2 + dz**2))

    @staticmethod
    def sort_render_order(elements: list[tuple]) -> list[tuple]:
        return sorted(elements, key=lambda e: e[0], reverse=True)
