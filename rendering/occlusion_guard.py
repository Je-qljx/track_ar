import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from detection.detector import Detection
from tracking.lane_assigner import AthleteState


@dataclass
class GraphicAnchor:
    world: WorldCoord
    image: ImageCoord
    offset_ahead: float
    placement_mode: str = "ahead"
    collision_resolved: bool = True


def compute_graphic_bbox(world: WorldCoord, projector: Projector, graphic_width_m: float = 1.0, graphic_height_m: float = 0.5) -> tuple[float, float, float, float]:
    corners_world = [
        WorldCoord(world.x - graphic_width_m / 2, world.y - graphic_height_m / 2, world.z),
        WorldCoord(world.x + graphic_width_m / 2, world.y - graphic_height_m / 2, world.z),
        WorldCoord(world.x + graphic_width_m / 2, world.y + graphic_height_m / 2, world.z),
        WorldCoord(world.x - graphic_width_m / 2, world.y + graphic_height_m / 2, world.z),
    ]
    corners_img = [projector.project(c) for c in corners_world]
    us = [c.u for c in corners_img]
    vs = [c.v for c in corners_img]
    return (min(us), min(vs), max(us), max(vs))


DEFAULT_OFFSET_AHEAD = 2.0
MAX_OFFSET_AHEAD = 5.0
OFFSET_BEHIND = 1.0
MAX_OFFSET_BEHIND = 3.0
LATERAL_SHIFT = 0.4
VERTICAL_LIFT = 0.3
GRAPHIC_WIDTH = 0.8 * 1.22
GRAPHIC_HEIGHT = 0.4

_PLACEMENT_CANDIDATES = [
    (2.0, 0.0, "ahead"),
    (4.0, 0.0, "ahead"),
    (1.0, 0.0, "behind"),
    (3.0, 0.0, "behind"),
    (2.0, -0.4, "ahead_lateral"),
    (2.0, 0.4, "ahead_lateral"),
]


class OcclusionGuard:
    def __init__(self, geometry: TrackGeometry, projector: Projector):
        self.geometry = geometry
        self.projector = projector

    def compute_safe_position(
        self,
        athlete: AthleteState,
        all_athletes: list[AthleteState],
        distance_to_end: float | None = None,
    ) -> GraphicAnchor:
        lane = athlete.lane
        d_m = athlete.d_m
        for ahead_m, lateral_m, mode in _PLACEMENT_CANDIDATES:
            g_dm = d_m + (ahead_m if not mode.startswith("behind") else -ahead_m)
            if distance_to_end is not None and g_dm > self.geometry.length:
                continue
            if g_dm < 0:
                continue
            G_world = self.geometry.world_coord(lane, g_dm, lateral_shift=lateral_m, z=0.0)
            G_img = self.projector.project(G_world)
            return GraphicAnchor(world=G_world, image=G_img, offset_ahead=ahead_m, placement_mode=mode)
        G_world = self.geometry.world_coord(lane, d_m, lateral_shift=-LATERAL_SHIFT, z=VERTICAL_LIFT)
        G_img = self.projector.project(G_world)
        return GraphicAnchor(world=G_world, image=G_img, offset_ahead=0, placement_mode="fallback")
